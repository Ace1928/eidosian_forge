import argparse
import copy
import functools
import logging
import os
import shutil
import sys
import textwrap
from importlib import import_module
from typing import Union
import torch
import torch.fx as fx
from torch._dynamo.debug_utils import (
from torch.fx.experimental.symbolic_shapes import fx_placeholder_targets
from torch.hub import tqdm
from .. import config
from ..backends.registry import lookup_backend, register_debug_backend
from ..debug_utils import clone_inputs_retaining_gradness
from math import inf
import torch
from torch import tensor, device
import torch.fx as fx
import torch._dynamo
from torch._dynamo.testing import rand_strided
from torch._dynamo.debug_utils import run_fwd_maybe_bwd
def common_flags(parser):
    accuracy_group = parser.add_mutually_exclusive_group()
    accuracy_group.add_argument('--no-accuracy', dest='accuracy', action='store_const', const='', default=accuracy, help='do not test accuracy, just run the module and see if it errors')
    accuracy_group.add_argument('--accuracy', action='store_const', const='accuracy', default=accuracy, help='test accuracy')
    parser.add_argument('--save-dir', type=str, default=save_dir, metavar='DIR', help='directory where saved inputs live')
    parser.add_argument('--no-save-dir', dest='save_dir', action='store_const', const=None, help="don't use any directory for saved inputs")
    parser.add_argument('--no-isolate', dest='isolate', action='store_false', default=False, help="no isolate (doesn't do anything for after_dynamo)")
    parser.add_argument('--autocast', default=autocast, action='store_true', help='use torch.cuda.amp.autocast')
    parser.add_argument('--no-autocast', dest='autocast', action='store_false', help="don't use torch.cuda.amp.autocast")
    parser.add_argument('--backend', type=str, default=backend, metavar='BACKEND', help='torch.compile backend to use')