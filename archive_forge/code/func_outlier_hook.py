import json
import shlex
import subprocess
from typing import Tuple
import torch
def outlier_hook(module, input):
    assert isinstance(module, torch.nn.Linear)
    tracer = OutlierTracer.get_instance()
    hvalue = tracer.get_hvalue(module.weight)
    if hvalue not in tracer.hvalue2outlier_idx:
        outlier_idx = find_outlier_dims(module.weight)
        tracer.outliers.append(outlier_idx)
        tracer.hvalues.append(hvalue)
        if len(tracer.outliers) > 1:
            if tracer.outliers[-1].numel() > 0:
                assert tracer.outliers[-1].max() < module.weight.shape[1]
            tracer.hvalue2outlier_idx[hvalue] = tracer.outliers[-1]
        else:
            merged = input[0].view(-1, input[0].shape[-1])
            outlier_idx = find_outlier_dims(merged, reduction_dim=1, zscore=3)
            dims = (torch.abs(input[0]) > 6).sum(dim=list(range(len(input[0].shape) - 1)))
            outlier_idx2 = torch.where(dims > 0)[0]
            outlier_idx = torch.cat([outlier_idx, outlier_idx2]).unique()
            tracer.hvalue2outlier_idx[hvalue] = outlier_idx
    else:
        for hook in tracer.hooks:
            hook.remove()