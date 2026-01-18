from typing import cast, Dict, Hashable, Iterable, List, Optional, Sequence
from collections import OrderedDict
import dataclasses
import numpy as np
import cirq
from cirq_google.api import v2
from cirq_google.api.v2 import result_pb2
def results_from_proto(msg: result_pb2.Result, measurements: Optional[List[MeasureInfo]]=None) -> Sequence[Sequence[cirq.Result]]:
    """Converts a v2 result proto into List of list of trial results.

    Args:
        msg: v2 Result message to convert.
        measurements: List of info about expected measurements in the program.
            This may be used for custom ordering of the result. If no
            measurement config is provided, then all results will be returned
            in the order specified within the result.

    Returns:
        A list containing a list of trial results for each sweep.
    """
    measure_map = {m.key: m for m in measurements} if measurements else None
    return [_trial_sweep_from_proto(sweep_result, measure_map) for sweep_result in msg.sweep_results]