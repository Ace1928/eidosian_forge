import collections
import collections.abc
import math
import operator
import unittest
from dataclasses import asdict, dataclass
from enum import Enum
from functools import partial
from itertools import product
from typing import Any, Callable, Iterable, List, Optional, Tuple
from torchgen.utils import dataclass_repr
import torch
from torch.testing import make_tensor
from torch.testing._internal.common_device_type import (
from torch.testing._internal.common_dtype import (
from torch.testing._internal.common_utils import (
from torch.testing._internal.opinfo import utils
def sample_inputs(self, device, dtype, requires_grad=False, **kwargs):
    """
        Returns an iterable of SampleInputs.

        These samples should be sufficient to test the function works correctly
        with autograd, TorchScript, etc.
        """
    samples = self.sample_inputs_func(self, device, dtype, requires_grad, **kwargs)
    if kwargs.get('include_conjugated_inputs', False):
        conj_samples = self.conjugate_sample_inputs(device, dtype, requires_grad, **kwargs)
        samples_list = list(samples)
        samples_list.extend(conj_samples)
        samples = tuple(samples_list)
    return TrackedInputIter(iter(samples), 'sample input')