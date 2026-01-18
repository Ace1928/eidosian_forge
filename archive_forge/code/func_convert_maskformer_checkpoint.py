import argparse
import json
import pickle
from pathlib import Path
import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import MaskFormerConfig, MaskFormerForInstanceSegmentation, MaskFormerImageProcessor, ResNetConfig
from transformers.utils import logging
@torch.no_grad()
def convert_maskformer_checkpoint(model_name: str, checkpoint_path: str, pytorch_dump_folder_path: str, push_to_hub: bool=False):
    """
    Copy/paste/tweak model's weights to our MaskFormer structure.
    """
    config = get_maskformer_config(model_name)
    with open(checkpoint_path, 'rb') as f:
        data = pickle.load(f)
    state_dict = data['model']
    rename_keys = create_rename_keys(config)
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)
    read_in_decoder_q_k_v(state_dict, config)
    for key, value in state_dict.items():
        state_dict[key] = torch.from_numpy(value)
    model = MaskFormerForInstanceSegmentation(config)
    model.eval()
    model.load_state_dict(state_dict)
    image = prepare_img()
    if 'vistas' in model_name:
        ignore_index = 65
    elif 'cityscapes' in model_name:
        ignore_index = 65535
    else:
        ignore_index = 255
    reduce_labels = True if 'ade' in model_name else False
    image_processor = MaskFormerImageProcessor(ignore_index=ignore_index, reduce_labels=reduce_labels)
    inputs = image_processor(image, return_tensors='pt')
    outputs = model(**inputs)
    if model_name == 'maskformer-resnet50-ade':
        expected_logits = torch.tensor([[6.771, -0.1452, -3.5687], [1.9165, -1.001, -1.8614], [3.6209, -0.295, -1.3813]])
    elif model_name == 'maskformer-resnet101-ade':
        expected_logits = torch.tensor([[4.0381, -1.1483, -1.9688], [2.7083, -1.9147, -2.2555], [3.4367, -1.3711, -2.1609]])
    elif model_name == 'maskformer-resnet50-coco-stuff':
        expected_logits = torch.tensor([[3.2309, -3.0481, -2.8695], [5.4986, -5.4242, -2.4211], [6.21, -5.2279, -2.7786]])
    elif model_name == 'maskformer-resnet101-coco-stuff':
        expected_logits = torch.tensor([[4.7188, -3.2585, -2.8857], [6.6871, -2.9181, -1.2487], [7.2449, -2.2764, -2.1874]])
    elif model_name == 'maskformer-resnet101-cityscapes':
        expected_logits = torch.tensor([[-1.8861, -1.5465, 0.6749], [-2.3677, -1.6707, -0.0867], [-2.2314, -1.953, -0.9132]])
    elif model_name == 'maskformer-resnet50-vistas':
        expected_logits = torch.tensor([[-6.3917, -1.5216, -1.1392], [-5.5335, -4.5318, -1.8339], [-4.3576, -4.0301, 0.2162]])
    elif model_name == 'maskformer-resnet50-ade20k-full':
        expected_logits = torch.tensor([[3.6146, -1.9367, -3.2534], [4.0099, 0.2027, -2.7576], [3.3913, -2.3644, -3.9519]])
    elif model_name == 'maskformer-resnet101-ade20k-full':
        expected_logits = torch.tensor([[3.2211, -1.655, -2.7605], [2.8559, -2.4512, -2.9574], [2.6331, -2.6775, -2.1844]])
    assert torch.allclose(outputs.class_queries_logits[0, :3, :3], expected_logits, atol=0.0001)
    print('Looks ok!')
    if pytorch_dump_folder_path is not None:
        print(f'Saving model and image processor of {model_name} to {pytorch_dump_folder_path}')
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        model.save_pretrained(pytorch_dump_folder_path)
        image_processor.save_pretrained(pytorch_dump_folder_path)
    if push_to_hub:
        print(f'Pushing model and image processor of {model_name} to the hub...')
        model.push_to_hub(f'facebook/{model_name}')
        image_processor.push_to_hub(f'facebook/{model_name}')