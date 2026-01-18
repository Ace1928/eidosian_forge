import argparse
import numpy as np
import torch
from transformers import SpeechT5HifiGan, SpeechT5HifiGanConfig, logging
@torch.no_grad()
def convert_hifigan_checkpoint(checkpoint_path, stats_path, pytorch_dump_folder_path, config_path=None, repo_id=None):
    if config_path is not None:
        config = SpeechT5HifiGanConfig.from_pretrained(config_path)
    else:
        config = SpeechT5HifiGanConfig()
    model = SpeechT5HifiGan(config)
    orig_checkpoint = torch.load(checkpoint_path)
    load_weights(orig_checkpoint['model']['generator'], model, config)
    stats = np.load(stats_path)
    mean = stats[0].reshape(-1)
    scale = stats[1].reshape(-1)
    model.mean = torch.from_numpy(mean).float()
    model.scale = torch.from_numpy(scale).float()
    model.save_pretrained(pytorch_dump_folder_path)
    if repo_id:
        print('Pushing to the hub...')
        model.push_to_hub(repo_id)