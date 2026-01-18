import filelock
import glob
import fnmatch
import json
import os
from collections import defaultdict
from typing import Any, Iterator, List, Optional, Tuple
from huggingface_hub import snapshot_download, HfFileSystem
import numpy as np
from safetensors.torch import load_file, save_file, safe_open
import torch
from tqdm.auto import tqdm
from vllm.config import ModelConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization import (get_quantization_config,
def convert_pyslice_to_tensor(x: Any) -> torch.Tensor:
    """convert PySafeSlice object from safetensors to torch.Tensor

    PySafeSlice object supports indexing, which is done before loading the
    actual tensor and can reduce the amount of memory being read into the
    memory. However, it does not support more advanced functionalities
    like `.view()` or `.t()`. Therefore, if we need to modify the loaded
    tensor with these more complicated operators, we need to convert to
    tensor first.
    """
    if not isinstance(x, torch.Tensor):
        x = x[:]
    return x