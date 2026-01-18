import torch.fx as fx
import copy
import torch
import math
import sys
from typing import Callable, List
from functools import wraps, partial
from dataclasses import dataclass
from .compile_utils import get_placeholders, get_outputs
from torch.utils._content_store import ContentStoreWriter
from torch.hub import tqdm
from torch.multiprocessing.reductions import StorageWeakRef
import os
def create_minified_hlo_graph(minified_fx_graph, inputs):
    """
    Takes minified FX graph as primary input, and ports it to HLO via StableHLO
    Provides minified HLO graph as output, and archive them to local directory
    """
    hlo_dir = f'{os.getcwd()}/hlo_files'
    os.makedirs(hlo_dir, exists_ok=True)
    from torch_xla.stablehlo import save_torch_model_as_stablehlo
    save_torch_model_as_stablehlo(minified_fx_graph, inputs, hlo_dir)