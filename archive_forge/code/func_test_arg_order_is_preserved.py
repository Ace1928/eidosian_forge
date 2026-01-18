import inspect
import pickle
from types import GeneratorType
from typing import Any, Callable, Dict, Iterable, List, Optional, Union
import catalogue
import numpy
import pytest
import thinc.config
from thinc.api import Config, Model, NumpyOps, RAdam
from thinc.config import ConfigValidationError
from thinc.types import Generator, Ragged
from thinc.util import partial
from .util import make_tempdir
def test_arg_order_is_preserved():
    str_cfg = '\n    [model]\n\n    [model.chain]\n    @layers = "chain.v1"\n\n    [model.chain.*.hashembed]\n    @layers = "HashEmbed.v1"\n    nO = 8\n    nV = 8\n\n    [model.chain.*.expand_window]\n    @layers = "expand_window.v1"\n    window_size = 1\n    '
    cfg = Config().from_str(str_cfg)
    resolved = my_registry.resolve(cfg)
    model = resolved['model']['chain']
    assert model.name == 'hashembed>>expand_window'