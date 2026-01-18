import math
from types import MethodType
from typing import Any, Dict, List, Optional, Tuple, Union
from .state import PartialState
from .utils import (
def build_pipeline(model, split_points, args, kwargs, num_chunks):
    """
    Attaches the split points to the model based on `self.device_map` and generates a `PipelineStage`. Requires passing
    in needed `args` and `kwargs` as the model needs on the CPU.

    Users can pass in custom `num_chunks` as an optional hyper-parameter. By default will use
    `AcceleratorState.num_processes`
    """
    state = PartialState()
    annotate_split_points(model, {split_point: PipeSplitWrapper.SplitPoint.BEGINNING for split_point in split_points})
    found_batch_size = find_pippy_batch_size(args, kwargs)
    if found_batch_size != num_chunks:
        if args is not None:
            args = pad_input_tensors(args, found_batch_size, num_chunks)
        if kwargs is not None:
            kwargs = pad_input_tensors(kwargs, found_batch_size, num_chunks)
    pipe = Pipe.from_tracing(model, num_chunks=num_chunks, example_args=args, example_kwargs=kwargs)
    stage = PipelineStage(pipe, state.local_process_index, device=state.device)
    return stage