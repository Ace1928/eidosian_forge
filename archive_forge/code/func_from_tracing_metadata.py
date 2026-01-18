import collections
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, NewType, Optional, Set, Union
import torch
import torch.utils._pytree as pytree
from torch._guards import Source
from torch._subclasses import FakeTensor
from torch._subclasses.fake_tensor import is_fake
from .. import config
from .utils import strict_zip
@classmethod
def from_tracing_metadata(cls, *, in_spec: pytree.TreeSpec, out_spec: pytree.TreeSpec, graph_input_names: List[str], graph_output_names: List[str], view_mutation_metadata: ViewAndMutationMeta, named_parameters: List[str], named_buffers: List[str], num_user_inputs: int, num_user_outputs: int, loss_index: Optional[int], backward_signature: Optional[BackwardSignature]) -> 'GraphSignature':
    graph_inputs = graph_input_names
    graph_outputs = graph_output_names
    parameters = list(named_parameters)
    buffers = list(named_buffers)
    user_inputs = graph_inputs[len(parameters) + len(buffers):]
    assert num_user_inputs == len(user_inputs)
    assert len(graph_inputs) == len(parameters) + len(buffers) + len(user_inputs)
    inputs_to_parameters = dict(zip(graph_inputs[:len(parameters)], parameters))
    inputs_to_buffers = dict(zip(graph_inputs[len(parameters):len(parameters) + len(buffers)], buffers))
    state_names = [*parameters, *buffers]
    mutated_buffers = []
    for idx, input_info in enumerate(view_mutation_metadata.input_info):
        if input_info.mutates_data:
            assert idx >= len(parameters)
            buffer_name = state_names[idx]
            mutated_buffers.append(buffer_name)
    assert len(mutated_buffers) == view_mutation_metadata.num_mutated_inp_runtime_indices
    start, stop = (0, view_mutation_metadata.num_mutated_inp_runtime_indices)
    buffers_to_mutate = dict(zip(graph_outputs[start:stop], mutated_buffers))
    start, stop = (stop, stop + num_user_outputs)
    user_outputs = graph_outputs[start:stop]
    unused_outputs = len(graph_outputs) - stop
    if backward_signature is not None:
        unused_outputs -= len(backward_signature.gradients_to_parameters) + len(backward_signature.gradients_to_user_inputs)
    assert unused_outputs == 0
    return GraphSignature(parameters=parameters, buffers=buffers, user_inputs=user_inputs, user_outputs=user_outputs, inputs_to_buffers=inputs_to_buffers, inputs_to_parameters=inputs_to_parameters, buffers_to_mutate=buffers_to_mutate, in_spec=in_spec, out_spec=out_spec, backward_signature=backward_signature)