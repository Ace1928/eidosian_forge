import unittest
import torch
import pandas as pd
import concurrent.futures
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Callable, Dict, Any, Optional
def test_initialize_activation_types(self):
    activation_manager = IndegoActivation()
    self.assertEqual(len(activation_manager.activation_types), 6)
    self.assertIn('relu', activation_manager.activation_types)
    self.assertIn('sigmoid', activation_manager.activation_types)
    self.assertIn('tanh', activation_manager.activation_types)
    self.assertIn('leaky_relu', activation_manager.activation_types)
    self.assertIn('elu', activation_manager.activation_types)
    self.assertIn('gelu', activation_manager.activation_types)