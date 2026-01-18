import os.path
from glob import glob
from typing import cast
import torch
from torch.types import Storage
def serialization_id(self):
    if self.has_record(__serialization_id_record_name__):
        return self.get_record(__serialization_id_record_name__)
    else:
        return ''