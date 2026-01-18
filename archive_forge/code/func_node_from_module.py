import itertools
from functools import reduce
from operator import mul
from typing import List
import wandb
from wandb import util
from wandb.data_types import Node
@classmethod
def node_from_module(cls, nid, module):
    numpy = util.get_module('numpy', 'Could not import numpy')
    node = wandb.Node()
    node.id = nid
    node.child_parameters = 0
    for parameter in module.parameters():
        node.child_parameters += numpy.prod(parameter.size())
    node.class_name = type(module).__name__
    return node