import dataclasses
import datetime
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, TYPE_CHECKING, Union
import numpy as np
import sympy
from cirq import ops, protocols, value
from cirq._compat import proper_repr
from cirq.work.observable_settings import (
def flatten_grouped_results(grouped_results: List[BitstringAccumulator]) -> List[ObservableMeasuredResult]:
    """Flatten a collection of BitstringAccumulators into a list of ObservableMeasuredResult.

    Raw results are contained in BitstringAccumulator which contains
    structure related to how the observables were measured (i.e. their
    grouping). This can be important for taking covariances into account.
    This function removes that structure, giving a flat list of results
    which may be easier to work with.

    Args:
        grouped_results: A list of BitstringAccumulators, probably returned
            from `measure_observables` or `measure_grouped_settings`.
    """
    return [res for acc in grouped_results for res in acc.results]