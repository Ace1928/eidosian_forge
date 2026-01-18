import unittest
import torch
from trl import (
def require_diffusers(test_case):
    """
    Decorator marking a test that requires diffusers. Skips the test if diffusers is not available.
    """
    if not is_diffusers_available():
        test_case = unittest.skip('test requires diffusers')(test_case)
    return test_case