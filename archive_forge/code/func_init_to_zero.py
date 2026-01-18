import torch
from .. import Config, autotune, cdiv, heuristics, jit
from .. import language as tl
from .matmul_perf_model import early_config_prune, estimate_matmul_time
def init_to_zero(name):
    return lambda nargs: nargs[name].zero_()