import os
import torch
from ..logging import get_logger
from .constants import FSDP_MODEL_NAME, FSDP_PYTORCH_VERSION, OPTIMIZER_NAME
from .imports import is_torch_distributed_available
from .modeling import is_peft_model
from .versions import is_torch_version
def load_fsdp_model(fsdp_plugin, accelerator, model, input_dir, model_index=0, adapter_only=False):
    accelerator.wait_for_everyone()
    if fsdp_plugin.state_dict_type == StateDictType.FULL_STATE_DICT:
        is_multi_process = accelerator.num_processes > 1
        fsdp_plugin.state_dict_config.offload_to_cpu = is_multi_process
        fsdp_plugin.state_dict_config.rank0_only = is_multi_process
    with FSDP.state_dict_type(model, fsdp_plugin.state_dict_type, fsdp_plugin.state_dict_config, fsdp_plugin.optim_state_dict_config):
        if fsdp_plugin.state_dict_type == StateDictType.FULL_STATE_DICT:
            if type(model) != FSDP and accelerator.process_index != 0:
                if not fsdp_plugin.sync_module_states:
                    raise ValueError('Set the `sync_module_states` flag to `True` so that model states are synced across processes when initializing FSDP object')
                return
            weights_name = f'{FSDP_MODEL_NAME}.bin' if model_index == 0 else f'{FSDP_MODEL_NAME}_{model_index}.bin'
            input_model_file = os.path.join(input_dir, weights_name)
            logger.info(f'Loading model from {input_model_file}')
            state_dict = torch.load(input_model_file)
            logger.info(f'Model loaded from {input_model_file}')
        elif fsdp_plugin.state_dict_type == StateDictType.LOCAL_STATE_DICT:
            weights_name = f'{FSDP_MODEL_NAME}_rank{accelerator.process_index}.bin' if model_index == 0 else f'{FSDP_MODEL_NAME}_{model_index}_rank{accelerator.process_index}.bin'
            input_model_file = os.path.join(input_dir, weights_name)
            logger.info(f'Loading model from {input_model_file}')
            state_dict = torch.load(input_model_file)
            logger.info(f'Model loaded from {input_model_file}')
        elif fsdp_plugin.state_dict_type == StateDictType.SHARDED_STATE_DICT:
            ckpt_dir = os.path.join(input_dir, f'{FSDP_MODEL_NAME}_{model_index}') if f'{FSDP_MODEL_NAME}' not in input_dir else input_dir
            logger.info(f'Loading model from {ckpt_dir}')
            state_dict = {'model': _get_model_state_dict(model, adapter_only=adapter_only)}
            dist_cp.load_state_dict(state_dict=state_dict, storage_reader=dist_cp.FileSystemReader(ckpt_dir), planner=DefaultLoadPlanner())
            state_dict = state_dict['model']
            logger.info(f'Model loaded from {ckpt_dir}')
        load_result = _set_model_state_dict(model, state_dict, adapter_only=adapter_only)
    return load_result