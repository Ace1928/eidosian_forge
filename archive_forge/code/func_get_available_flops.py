from collections import deque
from typing import TYPE_CHECKING, Any, Callable, Deque, Dict, List, Optional, TypeVar, Union
import torch
from typing_extensions import override
from lightning_fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_1
from lightning_fabric.utilities.rank_zero import rank_zero_only, rank_zero_warn
def get_available_flops(device: torch.device, dtype: Union[torch.dtype, str]) -> Optional[int]:
    """Returns the available theoretical FLOPs.

    This is an optimistic upper limit that could only be achievable if only thick matmuls were run in a benchmark
    environment.

    """
    if device.type == 'cuda':
        device_name = torch.cuda.get_device_name(device)
        chip = device_name.lower()
        if 'h100' in chip:
            if 'hbm3' in chip:
                chip = 'h100 sxm'
            elif 'nvl' in chip:
                chip = 'h100 nvl'
            elif 'pcie' in chip or 'hbm2e' in chip:
                chip = 'h100 pcie'
        elif 'l4' in chip:
            chip = 'l40' if 'tesla' in chip else 'l4'
        elif 'geforce rtx' in chip:
            number = chip.split(' ')[3]
            extra = ''
            if 'super' in chip:
                extra = ' super'
            elif 'ti' in chip:
                extra = ' ti'
            chip = f'rtx {number}{extra}'
        elif 'a6000' in chip:
            chip = 'a6000'
        elif 'a100' in chip:
            chip = 'a100'
        elif 'a40' in chip:
            chip = 'a40'
        elif 'a10g' in chip:
            chip = 'a10g'
        elif 't4' in chip:
            chip = 't4'
        elif 'quadro rtx 5000' in chip:
            chip = 'quadro rtx 5000'
        elif 'titan rtx' in chip:
            chip = 'titan rtx'
        elif 'v100-sxm' in chip:
            chip = 'v100 sxm'
        elif 'v100-pcie' in chip:
            chip = 'v100 pcie'
        elif 'v100s-pcie' in chip:
            chip = 'v100s pcie'
        else:
            rank_zero_warn(f'FLOPs not found for {device_name!r}')
            return None
        if chip not in _CUDA_FLOPS:
            rank_zero_warn(f'FLOPs not found for {device_name!r}, chip is {chip!r}')
            return None
        dtype_to_flops = _CUDA_FLOPS[chip]
        if dtype is torch.float32:
            from lightning_fabric.accelerators.cuda import _is_ampere_or_later
            if _is_ampere_or_later() and torch.get_float32_matmul_precision() != 'highest':
                dtype = 'tfloat32'
        if dtype not in dtype_to_flops:
            rank_zero_warn(f'{device_name!r} does not support {dtype}')
            return None
        return int(dtype_to_flops[dtype])
    if device.type == 'xla':
        from lightning_fabric.accelerators.xla import _XLA_GREATER_EQUAL_2_1
        if _XLA_GREATER_EQUAL_2_1:
            from torch_xla._internal import tpu
        else:
            from torch_xla.experimental import tpu
        tpu_env = tpu.get_tpu_env()
        device_name = tpu_env.get('TYPE') or tpu_env['ACCELERATOR_TYPE'].split('-')[0]
        chip = device_name.lower()
        assert isinstance(device_name, str)
        if chip not in _TPU_FLOPS:
            rank_zero_warn(f'FLOPs not found for TPU {device_name!r} with {dtype}')
            return None
        return int(_TPU_FLOPS[chip])