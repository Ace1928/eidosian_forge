import os.path
from glob import glob
from typing import cast
import torch
from torch.types import Storage
def get_all_records(self):
    files = []
    for filename in glob(f'{self.directory}/**', recursive=True):
        if not os.path.isdir(filename):
            files.append(filename[len(self.directory) + 1:])
    return files