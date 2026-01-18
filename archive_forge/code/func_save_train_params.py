import os
import time
import yaml
from contextlib import nullcontext
from pathlib import Path
from pkg_resources import packaging
from datetime import datetime
import torch
import torch.cuda.nccl as nccl
import torch.distributed as dist
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from tqdm import tqdm
from transformers import LlamaTokenizer
import json
from llama_recipes.model_checkpointing import save_model_checkpoint, save_model_and_optimizer_sharded, save_optimizer_checkpoint
from llama_recipes.policies import fpSixteen,bfSixteen, get_llama_wrapper
from llama_recipes.utils.memory_utils import MemoryTrace
from accelerate.utils import is_xpu_available, is_ccl_available
def save_train_params(train_config, fsdp_config, rank):
    """
    This function saves the train_config and FSDP config into a train_params.yaml.
    This will be used by converter script in the inference folder to fetch the HF model name or path.
    It also would be hepful as a log for future references.
    """
    train_config_dict = {k: str(v) for k, v in vars(train_config).items() if not k.startswith('__')}
    fsdp_config_dict = {k: str(v) for k, v in vars(fsdp_config).items() if not k.startswith('__')}
    train_params_dict = {**train_config_dict, **fsdp_config_dict}
    folder_name = train_config.dist_checkpoint_root_folder + '/' + train_config.dist_checkpoint_folder + '-' + train_config.model_name
    save_dir = Path.cwd() / folder_name
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    config_yaml = yaml.dump(train_params_dict, indent=4)
    file_name = os.path.join(save_dir, 'train_params.yaml')
    if os.path.isdir(file_name):
        print(f'Error: {file_name} is a directory, not a file.')
    else:
        with open(file_name, 'w') as f:
            f.write(config_yaml)
        if rank == 0:
            print(f'training params are saved in {file_name}')