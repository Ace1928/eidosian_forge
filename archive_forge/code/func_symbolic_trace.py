import builtins
import collections
import functools
import inspect
import math
import operator
import os
import random
import warnings
from typing import Any, Callable, Dict, List, Optional, Type, Union
import torch
from torch import nn
from torch.fx import Graph, GraphModule, Proxy, Tracer
from torch.fx._compatibility import compatibility
from torch.fx.proxy import ParameterProxy
from .. import PretrainedConfig, PreTrainedModel, logging
from ..models.auto import get_values
from ..models.auto.modeling_auto import (
from ..pytorch_utils import is_torch_greater_or_equal_than_2_0
from ..utils import (
def symbolic_trace(model: PreTrainedModel, input_names: Optional[List[str]]=None, disable_check: bool=False, tracer_cls: Type[HFTracer]=HFTracer) -> GraphModule:
    """
    Performs symbolic tracing on the model.

    Args:
        model ([`PretrainedModel`]):
            The model to trace.
        input_names (`List[str]`, *optional*):
            The names of the inputs of the traced model. If unset, model.dummy_inputs.keys() are used instead.
        disable_check (`bool`, *optional*, defaults to `False`):
            If `True`, no check is done before trying to trace the model, this is mostly usesul for debugging purposes.
        tracer_cls (`Type[HFTracer]`, *optional*, defaults to `HFTracer`):
            The tracer class to use for instantiating the tracer. If unset, `HFTracer` is used instead.

    Returns:
        `torch.fx.GraphModule`: A GraphModule constructed by recording operations seen while tracing the model.

    Example:

        ```python
        from transformers.utils.fx import symbolic_trace

        traced_model = symbolic_trace(model, input_names=["input_ids", "attention_mask", "token_type_ids"])
        ```
    """
    if input_names is None:
        input_names = model.dummy_inputs.keys()
    input_names = list(input_names)
    concrete_args = get_concrete_args(model, input_names)
    if not disable_check:
        check_if_model_is_supported(model)
    tracer = tracer_cls()
    traced_graph = tracer.trace(model, concrete_args=concrete_args)
    traced = torch.fx.GraphModule(model, traced_graph)
    traced.config = model.config
    traced.class_for_deserialization = model.__class__
    traced.device = model.device
    return traced