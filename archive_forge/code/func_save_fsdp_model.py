import os
import torch
from ..logging import get_logger
from .constants import FSDP_MODEL_NAME, FSDP_PYTORCH_VERSION, OPTIMIZER_NAME
from .imports import is_torch_distributed_available
from .modeling import is_peft_model
from .versions import is_torch_version
def save_fsdp_model(fsdp_plugin, accelerator, model, output_dir, model_index=0, adapter_only=False):
    os.makedirs(output_dir, exist_ok=True)
    if fsdp_plugin.state_dict_type == StateDictType.FULL_STATE_DICT:
        is_multi_process = accelerator.num_processes > 1
        fsdp_plugin.state_dict_config.offload_to_cpu = is_multi_process
        fsdp_plugin.state_dict_config.rank0_only = is_multi_process
    with FSDP.state_dict_type(model, fsdp_plugin.state_dict_type, fsdp_plugin.state_dict_config, fsdp_plugin.optim_state_dict_config):
        state_dict = _get_model_state_dict(model, adapter_only=adapter_only)
        if fsdp_plugin.state_dict_type == StateDictType.FULL_STATE_DICT:
            weights_name = f'{FSDP_MODEL_NAME}.bin' if model_index == 0 else f'{FSDP_MODEL_NAME}_{model_index}.bin'
            output_model_file = os.path.join(output_dir, weights_name)
            if accelerator.process_index == 0:
                logger.info(f'Saving model to {output_model_file}')
                torch.save(state_dict, output_model_file)
                logger.info(f'Model saved to {output_model_file}')
        elif fsdp_plugin.state_dict_type == StateDictType.LOCAL_STATE_DICT:
            weights_name = f'{FSDP_MODEL_NAME}_rank{accelerator.process_index}.bin' if model_index == 0 else f'{FSDP_MODEL_NAME}_{model_index}_rank{accelerator.process_index}.bin'
            output_model_file = os.path.join(output_dir, weights_name)
            logger.info(f'Saving model to {output_model_file}')
            torch.save(state_dict, output_model_file)
            logger.info(f'Model saved to {output_model_file}')
        elif fsdp_plugin.state_dict_type == StateDictType.SHARDED_STATE_DICT:
            ckpt_dir = os.path.join(output_dir, f'{FSDP_MODEL_NAME}_{model_index}')
            os.makedirs(ckpt_dir, exist_ok=True)
            logger.info(f'Saving model to {ckpt_dir}')
            state_dict = {'model': state_dict}
            dist_cp.save_state_dict(state_dict=state_dict, storage_writer=dist_cp.FileSystemWriter(ckpt_dir), planner=DefaultSavePlanner())
            logger.info(f'Model saved to {ckpt_dir}')