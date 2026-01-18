import argparse
import json
import gdown
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import (
def get_videomae_config(model_name):
    config = VideoMAEConfig()
    set_architecture_configs(model_name, config)
    if 'finetuned' not in model_name:
        config.use_mean_pooling = False
    if 'finetuned' in model_name:
        repo_id = 'huggingface/label-files'
        if 'kinetics' in model_name:
            config.num_labels = 400
            filename = 'kinetics400-id2label.json'
        elif 'ssv2' in model_name:
            config.num_labels = 174
            filename = 'something-something-v2-id2label.json'
        else:
            raise ValueError("Model name should either contain 'kinetics' or 'ssv2' in case it's fine-tuned.")
        id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type='dataset'), 'r'))
        id2label = {int(k): v for k, v in id2label.items()}
        config.id2label = id2label
        config.label2id = {v: k for k, v in id2label.items()}
    return config