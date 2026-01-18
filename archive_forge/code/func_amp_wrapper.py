import torch
import torch.utils.benchmark as benchmark
def amp_wrapper(*inputs, **kwinputs):
    with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=amp):
        fn(*inputs, **kwinputs)