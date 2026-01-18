import argparse
import os
import re
import torch
from flax.traverse_util import flatten_dict
from t5x import checkpoints
from transformers import (
def get_flax_param(t5x_checkpoint_path):
    flax_params = checkpoints.load_t5x_checkpoint(t5x_checkpoint_path)
    flax_params = flatten_dict(flax_params)
    return flax_params