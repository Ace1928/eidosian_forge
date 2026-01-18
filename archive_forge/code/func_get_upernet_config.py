import argparse
import json
import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import ConvNextConfig, SegformerImageProcessor, UperNetConfig, UperNetForSemanticSegmentation
def get_upernet_config(model_name):
    auxiliary_in_channels = 384
    if 'tiny' in model_name:
        depths = [3, 3, 9, 3]
        hidden_sizes = [96, 192, 384, 768]
    if 'small' in model_name:
        depths = [3, 3, 27, 3]
        hidden_sizes = [96, 192, 384, 768]
    if 'base' in model_name:
        depths = [3, 3, 27, 3]
        hidden_sizes = [128, 256, 512, 1024]
        auxiliary_in_channels = 512
    if 'large' in model_name:
        depths = [3, 3, 27, 3]
        hidden_sizes = [192, 384, 768, 1536]
        auxiliary_in_channels = 768
    if 'xlarge' in model_name:
        depths = [3, 3, 27, 3]
        hidden_sizes = [256, 512, 1024, 2048]
        auxiliary_in_channels = 1024
    num_labels = 150
    repo_id = 'huggingface/label-files'
    filename = 'ade20k-id2label.json'
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type='dataset'), 'r'))
    id2label = {int(k): v for k, v in id2label.items()}
    label2id = {v: k for k, v in id2label.items()}
    backbone_config = ConvNextConfig(depths=depths, hidden_sizes=hidden_sizes, out_features=['stage1', 'stage2', 'stage3', 'stage4'])
    config = UperNetConfig(backbone_config=backbone_config, auxiliary_in_channels=auxiliary_in_channels, num_labels=num_labels, id2label=id2label, label2id=label2id)
    return config