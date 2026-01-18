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
def test_adaptive_activation_network_initialization6(self):
    activation_dict = ActivationDictionary()
    in_features = 15
    out_features = 12
    layers_info = [30, 45, 60, 75, 90, 105]
    network = AdaptiveActivationNetwork(activation_dict, in_features, out_features, layers_info)
    self.assertEqual(network.activations, activation_dict.activation_types)
    self.assertEqual(network.activation_keys, list(activation_dict.activation_types.keys()))
    self.assertEqual(len(network.model), len(layers_info) + 1)
    self.assertEqual(network.policy_network.input_dim, in_features + sum(layers_info))
    self.assertEqual(network.policy_network.output_dim, len(activation_dict.activation_types))