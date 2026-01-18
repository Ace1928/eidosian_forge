import os
import torch
from ..logging import get_logger
from .constants import FSDP_MODEL_NAME, FSDP_PYTORCH_VERSION, OPTIMIZER_NAME
from .imports import is_torch_distributed_available
from .modeling import is_peft_model
from .versions import is_torch_version
def load_fsdp_optimizer(fsdp_plugin, accelerator, optimizer, model, input_dir, optimizer_index=0, adapter_only=False):
    accelerator.wait_for_everyone()
    with FSDP.state_dict_type(model, fsdp_plugin.state_dict_type, fsdp_plugin.state_dict_config, fsdp_plugin.optim_state_dict_config):
        if fsdp_plugin.state_dict_type == StateDictType.FULL_STATE_DICT:
            optim_state = None
            if accelerator.process_index == 0 or not fsdp_plugin.optim_state_dict_config.rank0_only:
                optimizer_name = f'{OPTIMIZER_NAME}.bin' if optimizer_index == 0 else f'{OPTIMIZER_NAME}_{optimizer_index}.bin'
                input_optimizer_file = os.path.join(input_dir, optimizer_name)
                logger.info(f'Loading Optimizer state from {input_optimizer_file}')
                optim_state = torch.load(input_optimizer_file)
                logger.info(f'Optimizer state loaded from {input_optimizer_file}')
        else:
            ckpt_dir = os.path.join(input_dir, f'{OPTIMIZER_NAME}_{optimizer_index}') if f'{OPTIMIZER_NAME}' not in input_dir else input_dir
            logger.info(f'Loading Optimizer from {ckpt_dir}')
            optim_state = load_sharded_optimizer_state_dict(model_state_dict=_get_model_state_dict(model, adapter_only=adapter_only), optimizer_key='optimizer', storage_reader=dist_cp.FileSystemReader(ckpt_dir))
            optim_state = optim_state['optimizer']
            logger.info(f'Optimizer loaded from {ckpt_dir}')
        flattened_osd = FSDP.optim_state_dict_to_load(model=model, optim=optimizer, optim_state_dict=optim_state)
        optimizer.load_state_dict(flattened_osd)