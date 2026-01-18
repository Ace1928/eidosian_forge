import argparse
import os
from pathlib import Path
import torch
from accelerate.utils.modeling import find_tied_parameters
from seamless_communication.models.inference.translator import Translator
from transformers import (
from transformers.utils import logging
def param_count(model):
    return sum((p[1].numel() for p in model.named_parameters() if 'final_proj' not in p[0]))