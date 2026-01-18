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
def get_lock(model_name_or_path: str, cache_dir: Optional[str]=None):
    lock_dir = cache_dir if cache_dir is not None else '/tmp'
    lock_file_name = model_name_or_path.replace('/', '-') + '.lock'
    lock = filelock.FileLock(os.path.join(lock_dir, lock_file_name))
    return lock