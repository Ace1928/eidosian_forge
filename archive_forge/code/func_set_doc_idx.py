from lib2to3.pgen2 import token
import os
import torch
import numpy as np
import shutil
import struct
from functools import lru_cache
from itertools import accumulate
def set_doc_idx(self, doc_idx_):
    self._index._doc_idx = doc_idx_