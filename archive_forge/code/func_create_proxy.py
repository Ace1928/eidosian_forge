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
def create_proxy(self, kind, target, args, kwargs, name=None, type_expr=None, proxy_factory_fn=None):
    rv = super().create_proxy(kind, target, args, kwargs, name, type_expr, proxy_factory_fn)
    if kind == 'placeholder' and target in self.meta_args:
        rv.install_metadata(self.meta_args[target])
        return rv
    if target in self.orig_fns:
        if 'device' in kwargs:
            kwargs['device'] = 'meta'
    try:
        args_metas = torch.fx.node.map_aggregate(args, _proxies_to_metas)
        kwargs_metas = torch.fx.node.map_aggregate(kwargs, _proxies_to_metas)
        if kind == 'call_function':
            meta_target = _MANUAL_META_OVERRIDES.get(target, target)
            meta_out = meta_target(*args_metas, **kwargs_metas)
            if isinstance(meta_out, torch.Tensor):
                meta_out = meta_out.to(device='meta')
        elif kind == 'call_method':
            method = getattr(args_metas[0].__class__, target)
            meta_target = _MANUAL_META_OVERRIDES.get(method, method)
            meta_out = meta_target(*args_metas, **kwargs_metas)
        elif kind == 'call_module':
            if not hasattr(self, 'orig_forward'):
                raise AttributeError(f'{self} does not have an attribute called orig_forward')
            self._disable_module_getattr = True
            try:
                mod = self.root.get_submodule(target)
                mod_type = type(mod)
                if mod_type in _MANUAL_META_OVERRIDES:
                    meta_out = _MANUAL_META_OVERRIDES[mod_type](mod, *args_metas, **kwargs_metas)
                else:
                    meta_out = self.orig_forward(*args_metas, **kwargs_metas)
            finally:
                self._disable_module_getattr = False
        elif kind == 'get_attr':
            self._disable_module_getattr = True
            try:
                attr_itr = self.root
                atoms = target.split('.')
                for atom in atoms:
                    attr_itr = getattr(attr_itr, atom)
                if isinstance(attr_itr, torch.Tensor):
                    meta_out = attr_itr.to(device='meta')
                else:
                    meta_out = attr_itr
            finally:
                self._disable_module_getattr = False
        else:
            return rv
        if not isinstance(rv, Proxy):
            raise ValueError("Don't support composite output yet")
        rv.install_metadata(meta_out)
    except Exception as e:
        if _IS_IN_DEBUG_MODE:
            warnings.warn(f'Could not compute metadata for {kind} target {target}: {e}')
    return rv