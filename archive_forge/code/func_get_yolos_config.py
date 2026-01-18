import argparse
import json
from pathlib import Path
import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import YolosConfig, YolosForObjectDetection, YolosImageProcessor
from transformers.utils import logging
def get_yolos_config(yolos_name: str) -> YolosConfig:
    config = YolosConfig()
    if 'yolos_ti' in yolos_name:
        config.hidden_size = 192
        config.intermediate_size = 768
        config.num_hidden_layers = 12
        config.num_attention_heads = 3
        config.image_size = [800, 1333]
        config.use_mid_position_embeddings = False
    elif yolos_name == 'yolos_s_dWr':
        config.hidden_size = 330
        config.num_hidden_layers = 14
        config.num_attention_heads = 6
        config.intermediate_size = 1320
    elif 'yolos_s' in yolos_name:
        config.hidden_size = 384
        config.intermediate_size = 1536
        config.num_hidden_layers = 12
        config.num_attention_heads = 6
    elif 'yolos_b' in yolos_name:
        config.image_size = [800, 1344]
    config.num_labels = 91
    repo_id = 'huggingface/label-files'
    filename = 'coco-detection-id2label.json'
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type='dataset'), 'r'))
    id2label = {int(k): v for k, v in id2label.items()}
    config.id2label = id2label
    config.label2id = {v: k for k, v in id2label.items()}
    return config