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
def test_make_config_positional_args_dicts():
    cfg = {'hyper_params': {'n_hidden': 512, 'dropout': 0.2, 'learn_rate': 0.001}, 'model': {'@layers': 'chain.v1', '*': {'relu1': {'@layers': 'Relu.v1', 'nO': 512, 'dropout': 0.2}, 'relu2': {'@layers': 'Relu.v1', 'nO': 512, 'dropout': 0.2}, 'softmax': {'@layers': 'Softmax.v1'}}}, 'optimizer': {'@optimizers': 'Adam.v1', 'learn_rate': 0.001}}
    resolved = my_registry.resolve(cfg)
    model = resolved['model']
    X = numpy.ones((784, 1), dtype='f')
    model.initialize(X=X, Y=numpy.zeros((784, 1), dtype='f'))
    model.begin_update(X)
    model.finish_update(resolved['optimizer'])