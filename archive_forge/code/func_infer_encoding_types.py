from collections.abc import Mapping, MutableMapping
from copy import deepcopy
import json
import itertools
import re
import sys
import traceback
import warnings
from typing import (
from types import ModuleType
import jsonschema
import pandas as pd
import numpy as np
from pandas.api.types import infer_dtype
from altair.utils.schemapi import SchemaBase
from altair.utils._dfi_types import Column, DtypeKind, DataFrame as DfiDataFrame
from typing import Literal, Protocol, TYPE_CHECKING, runtime_checkable
def infer_encoding_types(args: Sequence, kwargs: MutableMapping, channels: ModuleType):
    """Infer typed keyword arguments for args and kwargs

    Parameters
    ----------
    args : Sequence
        Sequence of function args
    kwargs : MutableMapping
        Dict of function kwargs
    channels : ModuleType
        The module containing all altair encoding channel classes.

    Returns
    -------
    kwargs : dict
        All args and kwargs in a single dict, with keys and types
        based on the channels mapping.
    """
    channel_objs = (getattr(channels, name) for name in dir(channels))
    channel_objs = (c for c in channel_objs if isinstance(c, type) and issubclass(c, SchemaBase))
    channel_to_name: Dict[Type[SchemaBase], str] = {c: c._encoding_name for c in channel_objs}
    name_to_channel: Dict[str, Dict[str, Type[SchemaBase]]] = {}
    for chan, name in channel_to_name.items():
        chans = name_to_channel.setdefault(name, {})
        if chan.__name__.endswith('Datum'):
            key = 'datum'
        elif chan.__name__.endswith('Value'):
            key = 'value'
        else:
            key = 'field'
        chans[key] = chan
    for arg in args:
        if isinstance(arg, (list, tuple)) and len(arg) > 0:
            type_ = type(arg[0])
        else:
            type_ = type(arg)
        encoding = channel_to_name.get(type_, None)
        if encoding is None:
            raise NotImplementedError('positional of type {}'.format(type_))
        if encoding in kwargs:
            raise ValueError('encoding {} specified twice.'.format(encoding))
        kwargs[encoding] = arg

    def _wrap_in_channel_class(obj, encoding):
        if isinstance(obj, SchemaBase):
            return obj
        if isinstance(obj, str):
            obj = {'shorthand': obj}
        if isinstance(obj, (list, tuple)):
            return [_wrap_in_channel_class(subobj, encoding) for subobj in obj]
        if encoding not in name_to_channel:
            warnings.warn("Unrecognized encoding channel '{}'".format(encoding), stacklevel=1)
            return obj
        classes = name_to_channel[encoding]
        cls = classes['value'] if 'value' in obj else classes['field']
        try:
            return cls.from_dict(obj, validate=False)
        except jsonschema.ValidationError:
            return obj
    return {encoding: _wrap_in_channel_class(obj, encoding) for encoding, obj in kwargs.items()}