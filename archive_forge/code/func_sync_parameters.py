import builtins
import copy
import os
import pickle
import contextlib
import subprocess
import socket
import parlai.utils.logging as logging
def sync_parameters(model: torch.nn.Module) -> bool:
    """
    Sync all parameters across all workers are the same.

    Always returns True, or raises an AssertionError if there was a failure.

    :param model: A pytorch model.
    :return: always True
    """
    if not is_distributed():
        return True
    with torch.no_grad():
        for p in model.parameters():
            if not is_primary_worker():
                p.data.zero_()
            dist.all_reduce(p.data, dist.ReduceOp.SUM)
    norm2 = sum(((p.data ** 2).sum().float().item() for p in model.parameters()))
    all_versions = all_gather_list(norm2)
    if not all((n == norm2 for n in all_versions)):
        raise AssertionError('Some models parameters were out of sync. Got the following norms: {}'.format(' '.join((str(x) for x in all_versions))))
    return True