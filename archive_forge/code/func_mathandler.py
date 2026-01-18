import io
import json
import os.path
import pickle
import tempfile
import torch
from torch.utils.data.datapipes.utils.common import StreamWrapper
def mathandler(**loadmat_kwargs):
    return MatHandler(**loadmat_kwargs)