import copy
import dataclasses
import itertools
import os
from typing import Any, Callable, Dict, List
import torch
import torch._lazy as lazy
import torch._lazy.metrics as metrics
from torch import fx
from torch._lazy import computation, debug as lazy_debug
from torch._lazy.tensor_factory_functions import tensor_factory_functions
def extract_compiled_graph(model: fx.GraphModule, example_inputs) -> Callable:
    """
    Optimize an eager model with LTC and returns a wrapper to execute the
    compiled graph directly without retracing. It depends on other mechanisms
    like TorchDynamo guards to guarantee the returned wrapper is only called
    when it's safe.
    """
    lazy_args = [arg.to(device='lazy') for arg in example_inputs]
    args_tensor_ids = [lazy.get_tensor_id(lazy_arg) for lazy_arg in lazy_args]
    tensor_id_to_arg_idx = {tensor_id: i for i, tensor_id in enumerate(args_tensor_ids)}
    lazy_model = copy.deepcopy(model).to(device=torch.device('lazy'))
    force_lazy_device(lazy_model)
    metrics.reset()
    lazy_out = lazy_model(*lazy_args)
    fallback_ops = get_fallback_ops()
    metrics.reset()
    if len(fallback_ops) > 0:
        raise RuntimeError(f'Fail to extact the compiled graph because of fallback: {','.join(fallback_ops)}')
    if not isinstance(lazy_out, (tuple, list)):
        lazy_out = (lazy_out,)
    args_and_out = tuple(lazy_args) + tuple(lazy_out)
    return_value_handler = ReturnValueHandler(args_and_out)
    if debug:
        print('Fx code:\n', model.code)
        print('LTC IR:', lazy_debug.dump_ir(args_and_out, 'text'))
    graph_input_tensor_ids, graph_input_ivalues = computation.get_tensors_ts_device_data_node(args_and_out)
    assert len(graph_input_tensor_ids) == len(graph_input_ivalues)
    graph_input_matcher = GraphInputMatcher(tensor_id_to_arg_idx, graph_input_tensor_ids, graph_input_ivalues)
    graph_hash = computation.get_graph_hash(args_and_out)
    if debug:
        print('graph_hash', graph_hash)
        print(f'args_tensor_ids {args_tensor_ids}')
        print('tensor ids from device data:', graph_input_tensor_ids)
    lazy.sync_multi(args_and_out, [])

    def optimized_mod(*args):
        if len(args_and_out) == 0:
            return ()
        graph_input = graph_input_matcher(args)
        res = return_value_handler.duplicate_eager_tensors(computation.run_cached_graph(graph_hash, graph_input))
        assert len(res) == len(args_and_out)
        for i, arg in enumerate(args):
            if arg is not res[i]:
                arg.copy_(res[i])
        return res[len(args):]
    return optimized_mod