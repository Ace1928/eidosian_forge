import argparse
import json
from pathlib import Path
import torch
import torchaudio
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import ASTConfig, ASTFeatureExtractor, ASTForAudioClassification
from transformers.utils import logging
def get_audio_spectrogram_transformer_config(model_name):
    config = ASTConfig()
    if '10-10' in model_name:
        pass
    elif 'speech-commands' in model_name:
        config.max_length = 128
    elif '12-12' in model_name:
        config.time_stride = 12
        config.frequency_stride = 12
    elif '14-14' in model_name:
        config.time_stride = 14
        config.frequency_stride = 14
    elif '16-16' in model_name:
        config.time_stride = 16
        config.frequency_stride = 16
    else:
        raise ValueError('Model not supported')
    repo_id = 'huggingface/label-files'
    if 'speech-commands' in model_name:
        config.num_labels = 35
        filename = 'speech-commands-v2-id2label.json'
    else:
        config.num_labels = 527
        filename = 'audioset-id2label.json'
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type='dataset'), 'r'))
    id2label = {int(k): v for k, v in id2label.items()}
    config.id2label = id2label
    config.label2id = {v: k for k, v in id2label.items()}
    return config