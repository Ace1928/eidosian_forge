import warnings
import torch
from torch.overrides import get_default_nowrap_functions
def is_sparse_coo(self):
    return self.layout == torch.sparse_coo