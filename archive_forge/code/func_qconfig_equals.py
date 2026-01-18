from collections import namedtuple
from typing import Optional, Any, Union, Type
import torch
import torch.nn as nn
from torch.ao.quantization.fake_quantize import (
from .observer import (
import warnings
import copy
def qconfig_equals(q1: QConfigAny, q2: QConfigAny):
    """
    Returns `True` if `q1` equals `q2`, and `False` otherwise.
    """
    if q1 is None or q2 is None:
        return q1 == q2
    else:
        assert q1 is not None and q2 is not None
        try:
            activation_same = _obs_or_fq_ctr_equals(q1.activation, q2.activation)
            weight_same = _obs_or_fq_ctr_equals(q1.weight, q2.weight)
            return activation_same and weight_same
        except AttributeError:
            return q1 == q2