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
def get_hip_file_path(rel_filepath, is_pytorch_extension=False):
    """
    Returns the new name of the hipified file
    """
    assert not os.path.isabs(rel_filepath)
    if not is_pytorch_extension and (not is_out_of_place(rel_filepath)):
        return rel_filepath
    dirpath, filename = os.path.split(rel_filepath)
    root, ext = os.path.splitext(filename)
    if ext == '.cu':
        ext = '.hip'
    orig_filename = filename
    orig_dirpath = dirpath
    dirpath = dirpath.replace('cuda', 'hip')
    dirpath = dirpath.replace('CUDA', 'HIP')
    dirpath = dirpath.replace('THC', 'THH')
    root = root.replace('cuda', 'hip')
    root = root.replace('CUDA', 'HIP')
    if dirpath != 'caffe2/core':
        root = root.replace('THC', 'THH')
    if not is_pytorch_extension and dirpath == orig_dirpath:
        dirpath = os.path.join(dirpath, 'hip')
    if is_pytorch_extension and dirpath == orig_dirpath and (root + ext == orig_filename):
        root = root + '_hip'
    return os.path.join(dirpath, root + ext)