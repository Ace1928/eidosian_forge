import unittest
import torch
from trl import (
def require_no_wandb(test_case):
    """
    Decorator marking a test that requires no wandb. Skips the test if wandb is available.
    """
    return require_wandb(test_case, required=False)