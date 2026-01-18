import argparse
import json
from pathlib import Path
import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import DetrConfig, DetrForObjectDetection, DetrForSegmentation, DetrImageProcessor, ResNetConfig
from transformers.utils import logging
@torch.no_grad()
def convert_detr_checkpoint(model_name, pytorch_dump_folder_path=None, push_to_hub=False):
    """
    Copy/paste/tweak model's weights to our DETR structure.
    """
    config, is_panoptic = get_detr_config(model_name)
    model_name_to_original_name = {'detr-resnet-50': 'detr_resnet50', 'detr-resnet-101': 'detr_resnet101'}
    logger.info(f'Converting model {model_name}...')
    detr = torch.hub.load('facebookresearch/detr', model_name_to_original_name[model_name], pretrained=True).eval()
    state_dict = detr.state_dict()
    for src, dest in create_rename_keys(config):
        if is_panoptic:
            src = 'detr.' + src
        rename_key(state_dict, src, dest)
    read_in_q_k_v(state_dict, is_panoptic=is_panoptic)
    prefix = 'detr.model.' if is_panoptic else 'model.'
    for key in state_dict.copy().keys():
        if is_panoptic:
            if key.startswith('detr') and (not key.startswith('class_labels_classifier')) and (not key.startswith('bbox_predictor')):
                val = state_dict.pop(key)
                state_dict['detr.model' + key[4:]] = val
            elif 'class_labels_classifier' in key or 'bbox_predictor' in key:
                val = state_dict.pop(key)
                state_dict['detr.' + key] = val
            elif key.startswith('bbox_attention') or key.startswith('mask_head'):
                continue
            else:
                val = state_dict.pop(key)
                state_dict[prefix + key] = val
        elif not key.startswith('class_labels_classifier') and (not key.startswith('bbox_predictor')):
            val = state_dict.pop(key)
            state_dict[prefix + key] = val
    model = DetrForSegmentation(config) if is_panoptic else DetrForObjectDetection(config)
    model.load_state_dict(state_dict)
    model.eval()
    format = 'coco_panoptic' if is_panoptic else 'coco_detection'
    processor = DetrImageProcessor(format=format)
    encoding = processor(images=prepare_img(), return_tensors='pt')
    pixel_values = encoding['pixel_values']
    original_outputs = detr(pixel_values)
    outputs = model(pixel_values)
    assert torch.allclose(outputs.logits, original_outputs['pred_logits'], atol=0.001)
    assert torch.allclose(outputs.pred_boxes, original_outputs['pred_boxes'], atol=0.001)
    if is_panoptic:
        assert torch.allclose(outputs.pred_masks, original_outputs['pred_masks'], atol=0.0001)
    print('Looks ok!')
    if pytorch_dump_folder_path is not None:
        logger.info(f'Saving PyTorch model and image processor to {pytorch_dump_folder_path}...')
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        model.save_pretrained(pytorch_dump_folder_path)
        processor.save_pretrained(pytorch_dump_folder_path)
    if push_to_hub:
        logger.info('Uploading PyTorch model and image processor to the hub...')
        model.push_to_hub(f'nielsr/{model_name}')
        processor.push_to_hub(f'nielsr/{model_name}')