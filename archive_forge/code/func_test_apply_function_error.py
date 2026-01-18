import unittest
import torch
import pandas as pd
import concurrent.futures
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Callable, Dict, Any, Optional
def test_apply_function_error(self):
    activation_manager = IndegoActivation()
    input = torch.tensor(0.5)
    with self.assertRaises(ValueError):
        activation_manager.apply_function('invalid_activation', input)