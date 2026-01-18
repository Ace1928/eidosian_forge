import itertools
from functools import reduce
from operator import mul
from typing import List
import wandb
from wandb import util
from wandb.data_types import Node
def create_forward_hook(self, name, graph_idx):
    graph = self

    def after_forward_hook(module, input, output):
        if id(module) not in self._graph_hooks:
            return
        if not isinstance(output, tuple):
            output = (output,)
        parameters = [(pname, list(param.size())) for pname, param in module.named_parameters()]
        node = Node(id=id(module), name=name, class_name=str(module), output_shape=nested_shape(output), parameters=parameters, num_parameters=[reduce(mul, size, 1) for pname, size in parameters])
        graph.nodes_by_id[id(module)] = node
        for param in module.parameters():
            graph.nodes_by_id[id(param)] = node
        graph.add_node(node)
        if not graph.criterion_passed:
            if hasattr(output[0], 'grad_fn'):
                graph.criterion = output[0].grad_fn
            elif isinstance(output[0], list) and output[0] and hasattr(output[0][0], 'grad_fn'):
                graph.criterion = output[0][0].grad_fn
        self._graph_hooks -= {id(module)}
        if not self._graph_hooks:
            wandb.run.summary['graph_%i' % graph_idx] = self
    return after_forward_hook