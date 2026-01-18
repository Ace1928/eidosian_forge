import random
from pathlib import Path
from typing import List
import numpy as np
import torch
from safetensors.torch import load_file
from torch.cuda.amp import GradScaler
from .utils import (
from .logging import get_logger
from .state import PartialState
def save_accelerator_state(output_dir: str, model_states: List[dict], optimizers: list, schedulers: list, dataloaders: list, process_index: int, scaler: GradScaler=None, save_on_each_node: bool=False, safe_serialization: bool=True):
    """
    Saves the current states of the models, optimizers, scaler, and RNG generators to a given directory.

    <Tip>

    If `safe_serialization` is `True`, models will be saved with `safetensors` while the rest are saved using native
    `pickle`.

    </Tip>

    Args:
        output_dir (`str` or `os.PathLike`):
            The name of the folder to save all relevant weights and states.
        model_states (`List[torch.nn.Module]`):
            A list of model states
        optimizers (`List[torch.optim.Optimizer]`):
            A list of optimizer instances
        schedulers (`List[torch.optim.lr_scheduler._LRScheduler]`):
            A list of learning rate schedulers
        dataloaders (`List[torch.utils.data.DataLoader]`):
            A list of dataloader instances to save their sampler states
        process_index (`int`):
            The current process index in the Accelerator state
        scaler (`torch.cuda.amp.GradScaler`, *optional*):
            An optional gradient scaler instance to save
        save_on_each_node (`bool`, *optional*):
            Whether to save on every node, or only the main node.
        safe_serialization (`bool`, *optional*, defaults to `True`):
            Whether to save the model using `safetensors` or the traditional PyTorch way (that uses `pickle`).
    """
    output_dir = Path(output_dir)
    for i, state in enumerate(model_states):
        weights_name = WEIGHTS_NAME if not safe_serialization else SAFE_WEIGHTS_NAME
        if i > 0:
            weights_name = weights_name.replace('.', f'_{i}.')
        output_model_file = output_dir.joinpath(weights_name)
        save(state, output_model_file, save_on_each_node=save_on_each_node, safe_serialization=safe_serialization)
        logger.info(f'Model weights saved in {output_model_file}')
    for i, opt in enumerate(optimizers):
        state = opt.state_dict()
        optimizer_name = f'{OPTIMIZER_NAME}.bin' if i == 0 else f'{OPTIMIZER_NAME}_{i}.bin'
        output_optimizer_file = output_dir.joinpath(optimizer_name)
        save(state, output_optimizer_file, save_on_each_node=save_on_each_node, safe_serialization=False)
        logger.info(f'Optimizer state saved in {output_optimizer_file}')
    for i, scheduler in enumerate(schedulers):
        state = scheduler.state_dict()
        scheduler_name = f'{SCHEDULER_NAME}.bin' if i == 0 else f'{SCHEDULER_NAME}_{i}.bin'
        output_scheduler_file = output_dir.joinpath(scheduler_name)
        save(state, output_scheduler_file, save_on_each_node=save_on_each_node, safe_serialization=False)
        logger.info(f'Scheduler state saved in {output_scheduler_file}')
    for i, dataloader in enumerate(dataloaders):
        sampler_name = f'{SAMPLER_NAME}.bin' if i == 0 else f'{SAMPLER_NAME}_{i}.bin'
        output_sampler_file = output_dir.joinpath(sampler_name)
        from .data_loader import IterableDatasetShard, SeedableRandomSampler
        if isinstance(dataloader.dataset, IterableDatasetShard):
            sampler = dataloader.sampler.sampler
            if isinstance(sampler, SeedableRandomSampler):
                save(sampler, output_sampler_file, save_on_each_node=save_on_each_node, safe_serialization=False)
        logger.info(f'Sampler state for dataloader {i} saved in {output_sampler_file}')
    if scaler is not None:
        state = scaler.state_dict()
        output_scaler_file = output_dir.joinpath(SCALER_NAME)
        torch.save(state, output_scaler_file)
        logger.info(f'Gradient scaler state saved in {output_scaler_file}')
    states = {}
    states_name = f'{RNG_STATE_NAME}_{process_index}.pkl'
    states['random_state'] = random.getstate()
    states['numpy_random_seed'] = np.random.get_state()
    states['torch_manual_seed'] = torch.get_rng_state()
    if is_xpu_available():
        states['torch_xpu_manual_seed'] = torch.xpu.get_rng_state_all()
    else:
        states['torch_cuda_manual_seed'] = torch.cuda.get_rng_state_all()
    if is_torch_xla_available():
        states['xm_seed'] = xm.get_rng_state()
    output_states_file = output_dir.joinpath(states_name)
    torch.save(states, output_states_file)
    logger.info(f'Random states saved in {output_states_file}')
    return output_dir