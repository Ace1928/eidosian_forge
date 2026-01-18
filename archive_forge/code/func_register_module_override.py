from collections import abc as container_abcs, defaultdict
from copy import deepcopy
from itertools import chain
import torch
import bitsandbytes.functional as F
def register_module_override(self, module, param_name, config):
    self.module_weight_config_triple.append((module, param_name, config))