import argparse
import json
from collections import OrderedDict
from pathlib import Path
import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import (
from transformers.utils import logging
@torch.no_grad()
def convert_conditional_detr_checkpoint(model_name, pytorch_dump_folder_path):
    """
    Copy/paste/tweak model's weights to our CONDITIONAL_DETR structure.
    """
    config = ConditionalDetrConfig()
    if 'resnet101' in model_name:
        config.backbone = 'resnet101'
    if 'dc5' in model_name:
        config.dilation = True
    is_panoptic = 'panoptic' in model_name
    if is_panoptic:
        config.num_labels = 250
    else:
        config.num_labels = 91
        repo_id = 'huggingface/label-files'
        filename = 'coco-detection-id2label.json'
        id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type='dataset'), 'r'))
        id2label = {int(k): v for k, v in id2label.items()}
        config.id2label = id2label
        config.label2id = {v: k for k, v in id2label.items()}
    format = 'coco_panoptic' if is_panoptic else 'coco_detection'
    image_processor = ConditionalDetrImageProcessor(format=format)
    img = prepare_img()
    encoding = image_processor(images=img, return_tensors='pt')
    pixel_values = encoding['pixel_values']
    logger.info(f'Converting model {model_name}...')
    conditional_detr = torch.hub.load('DeppMeng/ConditionalDETR', model_name, pretrained=True).eval()
    state_dict = conditional_detr.state_dict()
    for src, dest in rename_keys:
        if is_panoptic:
            src = 'conditional_detr.' + src
        rename_key(state_dict, src, dest)
    state_dict = rename_backbone_keys(state_dict)
    read_in_q_k_v(state_dict, is_panoptic=is_panoptic)
    prefix = 'conditional_detr.model.' if is_panoptic else 'model.'
    for key in state_dict.copy().keys():
        if is_panoptic:
            if key.startswith('conditional_detr') and (not key.startswith('class_labels_classifier')) and (not key.startswith('bbox_predictor')):
                val = state_dict.pop(key)
                state_dict['conditional_detr.model' + key[4:]] = val
            elif 'class_labels_classifier' in key or 'bbox_predictor' in key:
                val = state_dict.pop(key)
                state_dict['conditional_detr.' + key] = val
            elif key.startswith('bbox_attention') or key.startswith('mask_head'):
                continue
            else:
                val = state_dict.pop(key)
                state_dict[prefix + key] = val
        elif not key.startswith('class_labels_classifier') and (not key.startswith('bbox_predictor')):
            val = state_dict.pop(key)
            state_dict[prefix + key] = val
    model = ConditionalDetrForSegmentation(config) if is_panoptic else ConditionalDetrForObjectDetection(config)
    model.load_state_dict(state_dict)
    model.eval()
    model.push_to_hub(repo_id=model_name, organization='DepuMeng', commit_message='Add model')
    original_outputs = conditional_detr(pixel_values)
    outputs = model(pixel_values)
    assert torch.allclose(outputs.logits, original_outputs['pred_logits'], atol=0.0001)
    assert torch.allclose(outputs.pred_boxes, original_outputs['pred_boxes'], atol=0.0001)
    if is_panoptic:
        assert torch.allclose(outputs.pred_masks, original_outputs['pred_masks'], atol=0.0001)
    logger.info(f'Saving PyTorch model and image processor to {pytorch_dump_folder_path}...')
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    model.save_pretrained(pytorch_dump_folder_path)
    image_processor.save_pretrained(pytorch_dump_folder_path)