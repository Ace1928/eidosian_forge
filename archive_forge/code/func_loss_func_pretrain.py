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
def loss_func_pretrain(loss_mask, sentence_order, output_tensor):
    lm_loss_, sop_logits = output_tensor
    lm_loss_ = lm_loss_.float()
    loss_mask = loss_mask.float()
    lm_loss = torch.sum(lm_loss_.view(-1) * loss_mask.reshape(-1)) / loss_mask.sum()
    if sop_logits is not None:
        sop_loss = F.cross_entropy(sop_logits.view(-1, 2).float(), sentence_order.view(-1), ignore_index=-1)
        sop_loss = sop_loss.float()
        loss = lm_loss + sop_loss
        averaged_losses = average_losses_across_data_parallel_group([lm_loss, sop_loss])
        return (loss, {'lm loss': averaged_losses[0], 'sop loss': averaged_losses[1]})
    else:
        loss = lm_loss
        averaged_losses = average_losses_across_data_parallel_group([lm_loss])
        return (loss, {'lm loss': averaged_losses[0]})