from typing import Any, Callable, List, Optional, Union
from fugue import FugueWorkflow
from triad import assert_or_throw, conditional_dispatcher
from tune.concepts.dataset import TuneDataset, TuneDatasetBuilder
from tune.concepts.flow import Monitor
from tune.concepts.space import Space
from tune.concepts.logger import MetricLogger
from tune.constants import (
from tune.exceptions import TuneCompileError
from tune.iterative.objective import IterativeObjectiveFunc
from tune.noniterative.convert import to_noniterative_objective
from tune.noniterative.objective import (
from tune.noniterative.stopper import NonIterativeStopper
@conditional_dispatcher(entry_point=TUNE_PLUGINS)
def parse_logger(obj: Any) -> Union[MetricLogger, Callable[..., MetricLogger], None]:
    """On driver side parse an arbitrary object into a
    ``MetricLogger`` or a callable that generates a
    ``MetricLogger``.

    :param obj: the object.
    :return: ``MetricLogger`` or a callable that generates a
    ``MetricLogger`` or None
    """
    if obj is None or isinstance(obj, MetricLogger) or callable(obj):
        return obj
    raise TuneCompileError(f'{obj} is not a valid MetricLogger')