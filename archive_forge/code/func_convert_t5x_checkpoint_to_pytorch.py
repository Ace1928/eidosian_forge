import argparse
import collections
import numpy as np
import torch
from flax import traverse_util
from t5x import checkpoints
from transformers import MT5Config, UMT5EncoderModel, UMT5ForConditionalGeneration
from transformers.utils import logging
def convert_t5x_checkpoint_to_pytorch(t5x_checkpoint_path, config_file, pytorch_dump_path, is_encoder_only: bool=False, scalable_attention: bool=False):
    """Loads the config and model, converts the T5X checkpoint, and saves a PyTorch checkpoint."""
    config = MT5Config.from_json_file(config_file)
    print(f'Building PyTorch model from configuration: {config}')
    if is_encoder_only:
        model = UMT5EncoderModel(config)
    else:
        model = UMT5ForConditionalGeneration(config)
    load_t5x_weights_in_t5(model, config, t5x_checkpoint_path, is_encoder_only, scalable_attention)
    print(f'Save PyTorch model to {pytorch_dump_path}')
    model.save_pretrained(pytorch_dump_path)
    model.from_pretrained(pytorch_dump_path)
    print('Done')