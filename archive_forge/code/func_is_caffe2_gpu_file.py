import argparse
import fnmatch
import re
import shutil
import sys
import os
from . import constants
from .cuda_to_hip_mappings import CUDA_TO_HIP_MAPPINGS
from .cuda_to_hip_mappings import MATH_TRANSPILATIONS
from typing import Dict, List, Iterator, Optional
from collections.abc import Mapping, Iterable
from enum import Enum
def is_caffe2_gpu_file(rel_filepath):
    assert not os.path.isabs(rel_filepath)
    if rel_filepath.startswith('c10/cuda'):
        return True
    filename = os.path.basename(rel_filepath)
    _, ext = os.path.splitext(filename)
    return ('gpu' in filename or ext in ['.cu', '.cuh']) and 'cudnn' not in filename