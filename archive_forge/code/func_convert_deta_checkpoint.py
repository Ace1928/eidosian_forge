import argparse
import json
from pathlib import Path
import requests
import torch
from huggingface_hub import cached_download, hf_hub_download, hf_hub_url
from PIL import Image
from transformers import DetaConfig, DetaForObjectDetection, DetaImageProcessor
from transformers.utils import logging
@torch.no_grad()
def convert_deta_checkpoint(model_name, pytorch_dump_folder_path, push_to_hub):
    """
    Copy/paste/tweak model's weights to our DETA structure.
    """
    config = get_deta_config()
    if model_name == 'deta-resnet-50':
        filename = 'adet_checkpoint0011.pth'
    elif model_name == 'deta-resnet-50-24-epochs':
        filename = 'adet_2x_checkpoint0023.pth'
    else:
        raise ValueError(f'Model name {model_name} not supported')
    checkpoint_path = hf_hub_download(repo_id='nielsr/deta-checkpoints', filename=filename)
    state_dict = torch.load(checkpoint_path, map_location='cpu')['model']
    rename_keys = create_rename_keys(config)
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)
    read_in_decoder_q_k_v(state_dict, config)
    for key in state_dict.copy().keys():
        if 'transformer.decoder.class_embed' in key or 'transformer.decoder.bbox_embed' in key:
            val = state_dict.pop(key)
            state_dict[key.replace('transformer.decoder', 'model.decoder')] = val
        if 'input_proj' in key:
            val = state_dict.pop(key)
            state_dict['model.' + key] = val
        if 'level_embed' in key or 'pos_trans' in key or 'pix_trans' in key or ('enc_output' in key):
            val = state_dict.pop(key)
            state_dict[key.replace('transformer', 'model')] = val
    model = DetaForObjectDetection(config)
    model.load_state_dict(state_dict)
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    processor = DetaImageProcessor(format='coco_detection')
    img = prepare_img()
    encoding = processor(images=img, return_tensors='pt')
    pixel_values = encoding['pixel_values']
    outputs = model(pixel_values.to(device))
    if model_name == 'deta-resnet-50':
        expected_logits = torch.tensor([[-7.3978, -2.5406, -4.1668], [-8.2684, -3.9933, -3.8096], [-7.0515, -3.7973, -5.8516]])
        expected_boxes = torch.tensor([[0.5043, 0.4973, 0.9998], [0.2542, 0.5489, 0.4748], [0.549, 0.2765, 0.057]])
    elif model_name == 'deta-resnet-50-24-epochs':
        expected_logits = torch.tensor([[-7.1688, -2.4857, -4.8669], [-7.863, -3.8154, -4.2674], [-7.273, -4.1865, -5.5323]])
        expected_boxes = torch.tensor([[0.5021, 0.4971, 0.9994], [0.2546, 0.5486, 0.4731], [0.1686, 0.1986, 0.2142]])
    assert torch.allclose(outputs.logits[0, :3, :3], expected_logits.to(device), atol=0.0001)
    assert torch.allclose(outputs.pred_boxes[0, :3, :3], expected_boxes.to(device), atol=0.0001)
    print('Everything ok!')
    if pytorch_dump_folder_path:
        logger.info(f'Saving PyTorch model and processor to {pytorch_dump_folder_path}...')
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        model.save_pretrained(pytorch_dump_folder_path)
        processor.save_pretrained(pytorch_dump_folder_path)
    if push_to_hub:
        print('Pushing model and processor to hub...')
        model.push_to_hub(f'jozhang97/{model_name}')
        processor.push_to_hub(f'jozhang97/{model_name}')