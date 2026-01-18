import argparse
import json
import os
import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import ConvNextImageProcessor, ConvNextV2Config, ConvNextV2ForImageClassification
from transformers.image_utils import PILImageResampling
from transformers.utils import logging
@torch.no_grad()
def convert_convnextv2_checkpoint(checkpoint_url, pytorch_dump_folder_path, save_model, push_to_hub):
    """
    Copy/paste/tweak model's weights to our ConvNeXTV2 structure.
    """
    print('Downloading original model from checkpoint...')
    config, expected_shape = get_convnextv2_config(checkpoint_url)
    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url)['model']
    print('Converting model parameters...')
    for key in state_dict.copy().keys():
        val = state_dict.pop(key)
        state_dict[rename_key(key)] = val
    for key in state_dict.copy().keys():
        val = state_dict.pop(key)
        if not key.startswith('classifier'):
            key = 'convnextv2.' + key
        state_dict[key] = val
    model = ConvNextV2ForImageClassification(config)
    model.load_state_dict(state_dict)
    model.eval()
    preprocessor = convert_preprocessor(checkpoint_url)
    inputs = preprocessor(images=prepare_img(), return_tensors='pt')
    logits = model(**inputs).logits
    if checkpoint_url == 'https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_atto_1k_224_ema.pt':
        expected_logits = torch.tensor([-0.393, 0.1747, -0.5246, 0.4177, 0.4295])
    elif checkpoint_url == 'https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_femto_1k_224_ema.pt':
        expected_logits = torch.tensor([-0.1727, -0.5341, -0.7818, -0.4745, -0.6566])
    elif checkpoint_url == 'https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_pico_1k_224_ema.pt':
        expected_logits = torch.tensor([-0.0333, 0.1563, -0.9137, 0.1054, 0.0381])
    elif checkpoint_url == 'https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_nano_1k_224_ema.pt':
        expected_logits = torch.tensor([-0.1744, -0.1555, -0.0713, 0.095, -0.1431])
    elif checkpoint_url == 'https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_tiny_1k_224_ema.pt':
        expected_logits = torch.tensor([0.9996, 0.1966, -0.4386, -0.3472, 0.6661])
    elif checkpoint_url == 'https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_base_1k_224_ema.pt':
        expected_logits = torch.tensor([-0.2553, -0.6708, -0.1359, 0.2518, -0.2488])
    elif checkpoint_url == 'https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_large_1k_224_ema.pt':
        expected_logits = torch.tensor([-0.0673, -0.5627, -0.3753, -0.2722, 0.0178])
    elif checkpoint_url == 'https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_huge_1k_224_ema.pt':
        expected_logits = torch.tensor([-0.6377, -0.7458, -0.215, 0.1184, -0.0597])
    elif checkpoint_url == 'https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_nano_22k_224_ema.pt':
        expected_logits = torch.tensor([1.0799, 0.2322, -0.886, 1.0219, 0.6231])
    elif checkpoint_url == 'https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_nano_22k_384_ema.pt':
        expected_logits = torch.tensor([0.3766, 0.4917, -1.1426, 0.9942, 0.6024])
    elif checkpoint_url == 'https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_tiny_22k_224_ema.pt':
        expected_logits = torch.tensor([0.422, -0.6919, -0.4317, -0.2881, -0.6609])
    elif checkpoint_url == 'https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_tiny_22k_384_ema.pt':
        expected_logits = torch.tensor([0.1082, -0.8286, -0.5095, 0.4681, -0.8085])
    elif checkpoint_url == 'https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_base_22k_224_ema.pt':
        expected_logits = torch.tensor([-0.2419, -0.6221, 0.2176, -0.098, -0.7527])
    elif checkpoint_url == 'https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_base_22k_384_ema.pt':
        expected_logits = torch.tensor([0.0391, -0.4371, 0.3786, 0.1251, -0.2784])
    elif checkpoint_url == 'https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_large_22k_224_ema.pt':
        expected_logits = torch.tensor([-0.0504, 0.5636, -0.1729, -0.6507, -0.3949])
    elif checkpoint_url == 'https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_large_22k_384_ema.pt':
        expected_logits = torch.tensor([0.356, 0.9486, 0.3149, -0.2667, -0.5138])
    elif checkpoint_url == 'https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_huge_22k_384_ema.pt':
        expected_logits = torch.tensor([-0.2469, -0.455, -0.5853, -0.081, 0.0309])
    elif checkpoint_url == 'https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_huge_22k_512_ema.pt':
        expected_logits = torch.tensor([-0.309, 0.0802, -0.0682, -0.1979, -0.2826])
    else:
        raise ValueError(f'Unknown URL: {checkpoint_url}')
    assert torch.allclose(logits[0, :5], expected_logits, atol=0.001)
    assert logits.shape == expected_shape
    print('Model outputs match the original results!')
    if save_model:
        print('Saving model to local...')
        if not os.path.isdir(pytorch_dump_folder_path):
            os.mkdir(pytorch_dump_folder_path)
        model.save_pretrained(pytorch_dump_folder_path)
        preprocessor.save_pretrained(pytorch_dump_folder_path)
    model_name = 'convnextv2'
    if 'atto' in checkpoint_url:
        model_name += '-atto'
    if 'femto' in checkpoint_url:
        model_name += '-femto'
    if 'pico' in checkpoint_url:
        model_name += '-pico'
    if 'nano' in checkpoint_url:
        model_name += '-nano'
    elif 'tiny' in checkpoint_url:
        model_name += '-tiny'
    elif 'base' in checkpoint_url:
        model_name += '-base'
    elif 'large' in checkpoint_url:
        model_name += '-large'
    elif 'huge' in checkpoint_url:
        model_name += '-huge'
    if '22k' in checkpoint_url and '1k' not in checkpoint_url:
        model_name += '-22k'
    elif '22k' in checkpoint_url and '1k' in checkpoint_url:
        model_name += '-22k-1k'
    elif '1k' in checkpoint_url:
        model_name += '-1k'
    if '224' in checkpoint_url:
        model_name += '-224'
    elif '384' in checkpoint_url:
        model_name += '-384'
    elif '512' in checkpoint_url:
        model_name += '-512'
    if push_to_hub:
        print(f'Pushing {model_name} to the hub...')
        model.push_to_hub(model_name)
        preprocessor.push_to_hub(model_name)