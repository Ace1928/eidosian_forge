import argparse
from typing import Dict
import tensorflow as tf
import torch
from tqdm import tqdm
from transformers import BigBirdPegasusConfig, BigBirdPegasusForConditionalGeneration
def convert_bigbird_pegasus_ckpt_to_pytorch(ckpt_path: str, save_dir: str, config_update: dict):
    tf_weights = get_tf_weights_as_numpy(ckpt_path)
    torch_model = convert_bigbird_pegasus(tf_weights, config_update)
    torch_model.save_pretrained(save_dir)