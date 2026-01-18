from parlai.core.torch_ranker_agent import TorchRankerAgent
from parlai.utils.fp16 import fp16_optimizer_wrapper
from parlai.utils.torch import neginf
import torch
from torch.optim.optimizer import Optimizer
from torch.nn.utils import clip_grad_norm_
def get_bert_optimizer(models, type_optimization, learning_rate, fp16=False):
    """
    Optimizes the network with AdamWithDecay.
    """
    if type_optimization not in patterns_optimizer:
        print('Error. Type optimizer must be one of %s' % str(patterns_optimizer.keys()))
    parameters_with_decay = []
    parameters_with_decay_names = []
    parameters_without_decay = []
    parameters_without_decay_names = []
    no_decay = ['bias', 'gamma', 'beta']
    patterns = patterns_optimizer[type_optimization]
    for model in models:
        for n, p in model.named_parameters():
            if any((t in n for t in patterns)):
                if any((t in n for t in no_decay)):
                    parameters_without_decay.append(p)
                    parameters_without_decay_names.append(n)
                else:
                    parameters_with_decay.append(p)
                    parameters_with_decay_names.append(n)
    optimizer_grouped_parameters = [{'params': parameters_with_decay, 'weight_decay': 0.01}, {'params': parameters_without_decay, 'weight_decay': 0.0}]
    optimizer = AdamWithDecay(optimizer_grouped_parameters, lr=learning_rate)
    if fp16:
        optimizer = fp16_optimizer_wrapper(optimizer)
    return optimizer