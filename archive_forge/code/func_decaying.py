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
@thinc.registry.schedules('my_cool_repetitive_schedule.v1')
def decaying(base_rate: float, repeat: int) -> List[float]:
    return repeat * [base_rate]