from collections import abc as container_abcs, defaultdict
from copy import deepcopy
from itertools import chain
import torch
import bitsandbytes.functional as F
def fill_qmap(self):
    self.name2qmap['dynamic'] = F.create_dynamic_map(signed=True)
    self.name2qmap['udynamic'] = F.create_dynamic_map(signed=False)