from collections import namedtuple
from typing import Optional, Any, Union, Type
import torch
import torch.nn as nn
from torch.ao.quantization.fake_quantize import (
from .observer import (
import warnings
import copy
def get_default_qconfig(backend='x86', version=0):
    """
    Returns the default PTQ qconfig for the specified backend.

    Args:
      * `backend` (str): a string representing the target backend. Currently supports
        `x86` (default), `fbgemm`, `qnnpack` and `onednn`.

    Return:
        qconfig
    """
    supported_backends = ['fbgemm', 'x86', 'qnnpack', 'onednn']
    if backend not in supported_backends:
        raise AssertionError('backend: ' + str(backend) + f' not supported. backend must be one of {supported_backends}')
    if version == 0:
        if backend == 'fbgemm':
            qconfig = QConfig(activation=HistogramObserver.with_args(reduce_range=True), weight=default_per_channel_weight_observer)
        elif backend == 'qnnpack':
            qconfig = QConfig(activation=HistogramObserver.with_args(reduce_range=False), weight=default_weight_observer)
        elif backend == 'onednn':
            if not torch.cpu._is_cpu_support_vnni():
                warnings.warn('Default qconfig of oneDNN backend with reduce_range of false may have accuracy issues on CPU without Vector Neural Network Instruction support.')
            qconfig = QConfig(activation=HistogramObserver.with_args(reduce_range=False), weight=default_per_channel_weight_observer)
        elif backend == 'x86':
            qconfig = QConfig(activation=HistogramObserver.with_args(reduce_range=True), weight=default_per_channel_weight_observer)
        else:
            qconfig = default_qconfig
    else:
        raise AssertionError('Version number: ' + str(version) + ' in get_default_qconfig is not supported. Version number must be 0')
    return qconfig