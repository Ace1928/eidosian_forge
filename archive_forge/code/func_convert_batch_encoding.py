from typing import List, Optional, Union
import numpy as np
import tensorflow as tf
from .feature_extraction_utils import BatchFeature
from .tokenization_utils_base import BatchEncoding
from .utils import logging
def convert_batch_encoding(*args, **kwargs):
    if args and isinstance(args[0], (BatchEncoding, BatchFeature)):
        args = list(args)
        args[0] = dict(args[0])
    elif 'x' in kwargs and isinstance(kwargs['x'], (BatchEncoding, BatchFeature)):
        kwargs['x'] = dict(kwargs['x'])
    return (args, kwargs)