import copy
from typing import Any, Callable, Dict, Optional, Tuple, no_type_check
from fugue._utils.interfaceless import is_class_method
from triad import assert_or_throw
from triad.collections.function_wrapper import (
from triad.utils.convert import get_caller_global_local_vars, to_function
from tune.concepts.flow import Trial, TrialReport
from tune.exceptions import TuneCompileError
from tune.noniterative.objective import NonIterativeObjectiveFunc
@no_type_check
@staticmethod
def from_func(func: Callable, min_better: bool) -> '_NonIterativeObjectiveFuncWrapper':
    f = _NonIterativeObjectiveFuncWrapper(min_better=min_better)
    w = _NonIterativeObjectiveWrapper(func)
    f._func = w._func
    f._orig_input = w._orig_input
    f._output_f = w._rt.to_report
    return f