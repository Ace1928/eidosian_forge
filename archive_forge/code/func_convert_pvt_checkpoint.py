import argparse
from pathlib import Path
import requests
import torch
from PIL import Image
from transformers import PvtConfig, PvtForImageClassification, PvtImageProcessor
from transformers.utils import logging
@torch.no_grad()
def convert_pvt_checkpoint(pvt_size, pvt_checkpoint, pytorch_dump_folder_path):
    """
    Copy/paste/tweak model's weights to our PVT structure.
    """
    if pvt_size == 'tiny':
        config_path = 'Zetatech/pvt-tiny-224'
    elif pvt_size == 'small':
        config_path = 'Zetatech/pvt-small-224'
    elif pvt_size == 'medium':
        config_path = 'Zetatech/pvt-medium-224'
    elif pvt_size == 'large':
        config_path = 'Zetatech/pvt-large-224'
    else:
        raise ValueError(f"Available model's size: 'tiny', 'small', 'medium', 'large', but '{pvt_size}' was given")
    config = PvtConfig(name_or_path=config_path)
    state_dict = torch.load(pvt_checkpoint, map_location='cpu')
    rename_keys = create_rename_keys(config)
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)
    read_in_k_v(state_dict, config)
    model = PvtForImageClassification(config).eval()
    model.load_state_dict(state_dict)
    image_processor = PvtImageProcessor(size=config.image_size)
    encoding = image_processor(images=prepare_img(), return_tensors='pt')
    pixel_values = encoding['pixel_values']
    outputs = model(pixel_values)
    logits = outputs.logits.detach().cpu()
    if pvt_size == 'tiny':
        expected_slice_logits = torch.tensor([-1.4192, -1.9158, -0.9702])
    elif pvt_size == 'small':
        expected_slice_logits = torch.tensor([0.4353, -0.196, -0.2373])
    elif pvt_size == 'medium':
        expected_slice_logits = torch.tensor([-0.2914, -0.2231, 0.0321])
    elif pvt_size == 'large':
        expected_slice_logits = torch.tensor([0.374, -0.7739, -0.4214])
    else:
        raise ValueError(f"Available model's size: 'tiny', 'small', 'medium', 'large', but '{pvt_size}' was given")
    assert torch.allclose(logits[0, :3], expected_slice_logits, atol=0.0001)
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    print(f'Saving model pytorch_model.bin to {pytorch_dump_folder_path}')
    model.save_pretrained(pytorch_dump_folder_path)
    print(f'Saving image processor to {pytorch_dump_folder_path}')
    image_processor.save_pretrained(pytorch_dump_folder_path)