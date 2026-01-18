import gc
import unittest
import torch
from trl import is_diffusers_available, is_peft_available
from .testing_utils import require_diffusers
def scorer_function(images, prompts, metadata):
    return (torch.randn(1) * 3.0, {})