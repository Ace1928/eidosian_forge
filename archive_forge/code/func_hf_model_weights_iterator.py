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
def hf_model_weights_iterator(model_name_or_path: str, cache_dir: Optional[str]=None, load_format: str='auto', revision: Optional[str]=None, fall_back_to_pt: Optional[bool]=True) -> Iterator[Tuple[str, torch.Tensor]]:
    hf_folder, hf_weights_files, use_safetensors = prepare_hf_model_weights(model_name_or_path, cache_dir=cache_dir, load_format=load_format, fall_back_to_pt=fall_back_to_pt, revision=revision)
    if load_format == 'npcache':
        assert use_safetensors is False
        np_folder = os.path.join(hf_folder, 'np')
        os.makedirs(np_folder, exist_ok=True)
        weight_names_file = os.path.join(np_folder, 'weight_names.json')
        with get_lock(model_name_or_path, cache_dir):
            if not os.path.exists(weight_names_file):
                weight_names = []
                for bin_file in hf_weights_files:
                    state = torch.load(bin_file, map_location='cpu')
                    for name, param in state.items():
                        param_path = os.path.join(np_folder, name)
                        with open(param_path, 'wb') as f:
                            np.save(f, param.cpu().detach().numpy())
                        weight_names.append(name)
                with open(weight_names_file, 'w') as f:
                    json.dump(weight_names, f)
        with open(weight_names_file, 'r') as f:
            weight_names = json.load(f)
        for name in weight_names:
            param_path = os.path.join(np_folder, name)
            with open(param_path, 'rb') as f:
                param = np.load(f)
            yield (name, torch.from_numpy(param))
    elif use_safetensors:
        for st_file in hf_weights_files:
            with safe_open(st_file, framework='pt') as f:
                for name in f.keys():
                    param = f.get_tensor(name)
                    yield (name, param)
    else:
        for bin_file in hf_weights_files:
            state = torch.load(bin_file, map_location='cpu')
            for name, param in state.items():
                yield (name, param)
            del state
            torch.cuda.empty_cache()