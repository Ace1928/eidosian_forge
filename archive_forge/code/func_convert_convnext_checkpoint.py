import argparse
import json
from pathlib import Path
import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import ConvNextConfig, ConvNextForImageClassification, ConvNextImageProcessor
from transformers.utils import logging
@torch.no_grad()
def convert_convnext_checkpoint(checkpoint_url, pytorch_dump_folder_path):
    """
    Copy/paste/tweak model's weights to our ConvNext structure.
    """
    config, expected_shape = get_convnext_config(checkpoint_url)
    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url)['model']
    for key in state_dict.copy().keys():
        val = state_dict.pop(key)
        state_dict[rename_key(key)] = val
    for key in state_dict.copy().keys():
        val = state_dict.pop(key)
        if not key.startswith('classifier'):
            key = 'convnext.' + key
        state_dict[key] = val
    model = ConvNextForImageClassification(config)
    model.load_state_dict(state_dict)
    model.eval()
    size = 224 if '224' in checkpoint_url else 384
    image_processor = ConvNextImageProcessor(size=size)
    pixel_values = image_processor(images=prepare_img(), return_tensors='pt').pixel_values
    logits = model(pixel_values).logits
    if checkpoint_url == 'https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth':
        expected_logits = torch.tensor([-0.121, -0.6605, 0.1918])
    elif checkpoint_url == 'https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth':
        expected_logits = torch.tensor([-0.4473, -0.1847, -0.6365])
    elif checkpoint_url == 'https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth':
        expected_logits = torch.tensor([0.4525, 0.7539, 0.0308])
    elif checkpoint_url == 'https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_384.pth':
        expected_logits = torch.tensor([0.3561, 0.635, -0.0384])
    elif checkpoint_url == 'https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth':
        expected_logits = torch.tensor([0.4174, -0.0989, 0.1489])
    elif checkpoint_url == 'https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_384.pth':
        expected_logits = torch.tensor([0.2513, -0.1349, -0.1613])
    elif checkpoint_url == 'https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth':
        expected_logits = torch.tensor([1.298, 0.3631, -0.1198])
    elif checkpoint_url == 'https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth':
        expected_logits = torch.tensor([1.2963, 0.1227, 0.1723])
    elif checkpoint_url == 'https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth':
        expected_logits = torch.tensor([1.7956, 0.839, 0.282])
    elif checkpoint_url == 'https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_1k_224.pth':
        expected_logits = torch.tensor([-0.2822, -0.0502, -0.0878])
    elif checkpoint_url == 'https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_1k_384.pth':
        expected_logits = torch.tensor([-0.5672, -0.073, -0.4348])
    elif checkpoint_url == 'https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_1k_224.pth':
        expected_logits = torch.tensor([0.2681, 0.2365, 0.6246])
    elif checkpoint_url == 'https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_1k_384.pth':
        expected_logits = torch.tensor([-0.2642, 0.3931, 0.5116])
    elif checkpoint_url == 'https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_1k_224_ema.pth':
        expected_logits = torch.tensor([-0.6677, -0.1873, -0.8379])
    elif checkpoint_url == 'https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_1k_384_ema.pth':
        expected_logits = torch.tensor([-0.7749, -0.2967, -0.6444])
    else:
        raise ValueError(f'Unknown URL: {checkpoint_url}')
    assert torch.allclose(logits[0, :3], expected_logits, atol=0.001)
    assert logits.shape == expected_shape
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    print(f'Saving model to {pytorch_dump_folder_path}')
    model.save_pretrained(pytorch_dump_folder_path)
    print(f'Saving image processor to {pytorch_dump_folder_path}')
    image_processor.save_pretrained(pytorch_dump_folder_path)
    print('Pushing model to the hub...')
    model_name = 'convnext'
    if 'tiny' in checkpoint_url:
        model_name += '-tiny'
    elif 'small' in checkpoint_url:
        model_name += '-small'
    elif 'base' in checkpoint_url:
        model_name += '-base'
    elif 'xlarge' in checkpoint_url:
        model_name += '-xlarge'
    elif 'large' in checkpoint_url:
        model_name += '-large'
    if '224' in checkpoint_url:
        model_name += '-224'
    elif '384' in checkpoint_url:
        model_name += '-384'
    if '22k' in checkpoint_url and '1k' not in checkpoint_url:
        model_name += '-22k'
    if '22k' in checkpoint_url and '1k' in checkpoint_url:
        model_name += '-22k-1k'
    model.push_to_hub(repo_path_or_name=Path(pytorch_dump_folder_path, model_name), organization='nielsr', commit_message='Add model')