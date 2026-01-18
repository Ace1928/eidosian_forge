import argparse
import math
from abc import ABC
from functools import partial
import torch
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
from ..optimizer import AcceleratedOptimizer
from ..scheduler import AcceleratedScheduler
from .imports import is_megatron_lm_available, is_transformers_available
from .operations import recursively_apply, send_to_device
def log_eval_results(self):
    args = get_args()
    if args.tensorboard_dir is None or self.iteration == 0:
        return
    args = get_args()
    writer = get_tensorboard_writer()
    string = f'validation loss at iteration {self.iteration} | '
    for key in self.eval_total_loss_dict:
        if key.endswith('_num_iters'):
            continue
        value = self.eval_total_loss_dict[key] / self.eval_total_loss_dict[key + '_num_iters']
        string += f'{key} value: {value} | '
        ppl = math.exp(min(20, value.item()))
        if args.pretraining_flag:
            string += f'{key} PPL: {ppl} | '
        if writer:
            writer.add_scalar(f'{key} validation', value.item(), self.iteration)
            if args.pretraining_flag:
                writer.add_scalar(f'{key} validation ppl', ppl, self.iteration)
    length = len(string) + 1
    print_rank_last('-' * length)
    print_rank_last(string)
    print_rank_last('-' * length)
    self.eval_total_loss_dict = {}