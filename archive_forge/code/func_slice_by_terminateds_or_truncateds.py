import collections
from functools import partial
import itertools
import sys
from numbers import Number
from typing import Dict, Iterator, Set, Union
from typing import List, Optional
import numpy as np
import tree  # pip install dm_tree
from ray.rllib.utils.annotations import DeveloperAPI, ExperimentalAPI, PublicAPI
from ray.rllib.utils.compression import pack, unpack, is_compressed
from ray.rllib.utils.deprecation import Deprecated, deprecation_warning
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.torch_utils import convert_to_torch_tensor
from ray.rllib.utils.typing import (
from ray.util import log_once
def slice_by_terminateds_or_truncateds():
    slices = []
    offset = 0
    for i in range(self.count):
        if self[SampleBatch.TERMINATEDS][i] or (SampleBatch.TRUNCATEDS in self and self[SampleBatch.TRUNCATEDS][i]):
            slices.append(self[offset:i + 1])
            offset = i + 1
    if offset != self.count:
        slices.append(self[offset:])
    return slices