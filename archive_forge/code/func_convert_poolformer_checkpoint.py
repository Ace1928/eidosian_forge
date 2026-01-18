import argparse
import json
from collections import OrderedDict
from pathlib import Path
import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import PoolFormerConfig, PoolFormerForImageClassification, PoolFormerImageProcessor
from transformers.utils import logging
@torch.no_grad()
def convert_poolformer_checkpoint(model_name, checkpoint_path, pytorch_dump_folder_path):
    """
    Copy/paste/tweak model's weights to our PoolFormer structure.
    """
    config = PoolFormerConfig()
    repo_id = 'huggingface/label-files'
    size = model_name[-3:]
    config.num_labels = 1000
    filename = 'imagenet-1k-id2label.json'
    expected_shape = (1, 1000)
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type='dataset'), 'r'))
    id2label = {int(k): v for k, v in id2label.items()}
    config.id2label = id2label
    config.label2id = {v: k for k, v in id2label.items()}
    if size == 's12':
        config.depths = [2, 2, 6, 2]
        config.hidden_sizes = [64, 128, 320, 512]
        config.mlp_ratio = 4.0
        crop_pct = 0.9
    elif size == 's24':
        config.depths = [4, 4, 12, 4]
        config.hidden_sizes = [64, 128, 320, 512]
        config.mlp_ratio = 4.0
        crop_pct = 0.9
    elif size == 's36':
        config.depths = [6, 6, 18, 6]
        config.hidden_sizes = [64, 128, 320, 512]
        config.mlp_ratio = 4.0
        config.layer_scale_init_value = 1e-06
        crop_pct = 0.9
    elif size == 'm36':
        config.depths = [6, 6, 18, 6]
        config.hidden_sizes = [96, 192, 384, 768]
        config.mlp_ratio = 4.0
        config.layer_scale_init_value = 1e-06
        crop_pct = 0.95
    elif size == 'm48':
        config.depths = [8, 8, 24, 8]
        config.hidden_sizes = [96, 192, 384, 768]
        config.mlp_ratio = 4.0
        config.layer_scale_init_value = 1e-06
        crop_pct = 0.95
    else:
        raise ValueError(f'Size {size} not supported')
    image_processor = PoolFormerImageProcessor(crop_pct=crop_pct)
    image = prepare_img()
    pixel_values = image_processor(images=image, return_tensors='pt').pixel_values
    logger.info(f'Converting model {model_name}...')
    state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    state_dict = rename_keys(state_dict)
    model = PoolFormerForImageClassification(config)
    model.load_state_dict(state_dict)
    model.eval()
    image_processor = PoolFormerImageProcessor(crop_pct=crop_pct)
    pixel_values = image_processor(images=prepare_img(), return_tensors='pt').pixel_values
    outputs = model(pixel_values)
    logits = outputs.logits
    if size == 's12':
        expected_slice = torch.tensor([-0.3045, -0.6758, -0.4869])
    elif size == 's24':
        expected_slice = torch.tensor([0.4402, -0.1374, -0.8045])
    elif size == 's36':
        expected_slice = torch.tensor([-0.608, -0.5133, -0.5898])
    elif size == 'm36':
        expected_slice = torch.tensor([0.3952, 0.2263, -1.2668])
    elif size == 'm48':
        expected_slice = torch.tensor([0.1167, -0.0656, -0.3423])
    else:
        raise ValueError(f'Size {size} not supported')
    assert logits.shape == expected_shape
    assert torch.allclose(logits[0, :3], expected_slice, atol=0.01)
    logger.info(f'Saving PyTorch model and image processor to {pytorch_dump_folder_path}...')
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    model.save_pretrained(pytorch_dump_folder_path)
    print(f'Saving image processor to {pytorch_dump_folder_path}')
    image_processor.save_pretrained(pytorch_dump_folder_path)