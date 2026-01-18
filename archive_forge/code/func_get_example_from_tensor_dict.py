import os
import warnings
from dataclasses import asdict
from enum import Enum
from typing import List, Optional, Union
from ...tokenization_utils import PreTrainedTokenizer
from ...utils import is_tf_available, logging
from .utils import DataProcessor, InputExample, InputFeatures
def get_example_from_tensor_dict(self, tensor_dict):
    """See base class."""
    return InputExample(tensor_dict['idx'].numpy(), tensor_dict['sentence1'].numpy().decode('utf-8'), tensor_dict['sentence2'].numpy().decode('utf-8'), str(tensor_dict['label'].numpy()))