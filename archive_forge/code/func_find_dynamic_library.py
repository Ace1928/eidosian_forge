import glob
import os
import sys
from warnings import warn
import torch
def find_dynamic_library(folder, filename):
    for ext in ('so', 'dll', 'dylib'):
        yield from glob.glob(os.path.join(folder, '**', filename + ext))