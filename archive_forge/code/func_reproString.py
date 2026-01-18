import torch
import numpy as np
import argparse
from typing import Dict
def reproString(current_seed, args):
    repro_str = f'python {__file__}'
    if args.cuda_fuser:
        repro_str += ' --cuda-fuser'
    if args.legacy_fuser:
        repro_str += ' --legacy-fuser'
    if args.profiling_executor:
        repro_str += ' --profiling-executor'
    if args.fp16:
        repro_str += ' --fp16'
    if args.cpu:
        repro_str += ' --cpu'
    repro_str += ' --max-num-tensor {} --max-tensor-dim {} --max-tensor-size {} --depth-factor {} --seed {} --repro-run'.format(args.max_num_tensor, args.max_tensor_dim, args.max_tensor_size, args.depth_factor, current_seed)
    return repro_str