import torch
from xformers.benchmarks.utils import TestCase, bench_functions
from xformers.triton.softmax import log_softmax as triton_log_softmax
from xformers.triton.softmax import softmax as triton_softmax
def pytorch_log_fw_bw(x):
    y = torch.norm(torch.log_softmax(x, dim=-1))
    y.backward()