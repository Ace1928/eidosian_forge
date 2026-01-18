from collections import abc as container_abcs, defaultdict
from copy import deepcopy
from itertools import chain
import torch
import bitsandbytes.functional as F
def register_parameters(self, params):
    param_groups = list(params)
    if not isinstance(param_groups[0], dict):
        param_groups = [{'params': param_groups}]
    for group_index, group in enumerate(param_groups):
        for p_index, p in enumerate(group['params']):
            if id(p) in self.pid2config:
                self.index2config[group_index, p_index] = self.pid2config[id(p)]