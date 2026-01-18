import collections
import csv
import importlib
import json
import os
import pickle
import sys
import traceback
import types
import warnings
from abc import ABC, abstractmethod
from collections import UserDict
from contextlib import contextmanager
from os.path import abspath, exists
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from ..dynamic_module_utils import custom_object_save
from ..feature_extraction_utils import PreTrainedFeatureExtractor
from ..image_processing_utils import BaseImageProcessor
from ..modelcard import ModelCard
from ..models.auto.configuration_auto import AutoConfig
from ..tokenization_utils import PreTrainedTokenizer
from ..utils import (
def ensure_tensor_on_device(self, **inputs):
    """
        Ensure PyTorch tensors are on the specified device.

        Args:
            inputs (keyword arguments that should be `torch.Tensor`, the rest is ignored):
                The tensors to place on `self.device`.
            Recursive on lists **only**.

        Return:
            `Dict[str, torch.Tensor]`: The same as `inputs` but on the proper device.
        """
    return self._ensure_tensor_on_device(inputs, self.device)