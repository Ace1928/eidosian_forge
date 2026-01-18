import itertools
from functools import reduce
from operator import mul
from typing import List
import wandb
from wandb import util
from wandb.data_types import Node
@classmethod
def hook_torch(cls, model, criterion=None, graph_idx=0):
    wandb.termlog('logging graph, to disable use `wandb.watch(log_graph=False)`')
    graph = TorchGraph()
    graph.hook_torch_modules(model, criterion, graph_idx=graph_idx)
    return graph