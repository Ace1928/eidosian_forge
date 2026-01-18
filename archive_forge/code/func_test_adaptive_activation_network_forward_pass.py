import unittest  # Importing the unittest module for creating and running tests
import torch  # Importing the PyTorch library for tensor computations and neural network operations
import sys  # Importing the sys module to interact with the Python runtime environment
import asyncio  # Importing the asyncio module for writing single-threaded concurrent code using coroutines
import logging  # Importing the logging module to enable logging of messages of various severity levels
from unittest.mock import (
import numpy as np  # Importing the numpy library for numerical operations on arrays
import unittest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from sklearn.metrics import precision_score, recall_score, f1_score
from collections import defaultdict
import numpy as np
import asyncio
from typing import Dict, Any, List, cast, Tuple, Callable, Optional, Union
import sys
import torch.optim as optim
import os
from ActivationDictionary import (
from IndegoAdaptAct import (
from IndegoLogging import configure_logging
def test_adaptive_activation_network_forward_pass(self):
    activation_dict = ActivationDictionary()
    in_features = 5
    out_features = 3
    layers_info = [10, 15]
    network = AdaptiveActivationNetwork(activation_dict, in_features, out_features, layers_info)
    input_tensor = torch.randn(128, in_features)
    output_tensor = network(input_tensor)
    self.assertEqual(output_tensor.shape, (128, out_features))