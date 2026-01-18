import os.path
from glob import glob
from typing import cast
import torch
from torch.types import Storage
def has_record(self, path):
    full_path = os.path.join(self.directory, path)
    return os.path.isfile(full_path)