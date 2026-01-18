import abc
import collections
import dataclasses
import functools
import math
import re
from typing import (
import numpy as np
import pandas as pd
import cirq
from cirq.experiments.xeb_fitting import XEBPhasedFSimCharacterizationOptions
from cirq_google.api import v2
from cirq_google.engine import (
from cirq_google.ops import FSimGateFamily, SycamoreGate
def merge_matching_results(results: Iterable[PhasedFSimCalibrationResult]) -> Optional[PhasedFSimCalibrationResult]:
    """Merges a collection of results into a single result.

    Args:
        results: List of results to merge. They must be compatible with each other: all gate and
            options fields must be equal and every characterized pair must be present only in one of
            the characterizations.

    Returns:
        New PhasedFSimCalibrationResult that contains all the parameters from every result in
        results or None when the results list is empty.

    Raises:
        ValueError: If the gate and options fields are not all equal, or if the
            results have shared keys.
    """
    all_parameters: Dict[Tuple[cirq.Qid, cirq.Qid], PhasedFSimCharacterization] = {}
    common_gate = None
    common_options = None
    for result in results:
        if common_gate is None:
            common_gate = result.gate
        elif common_gate != result.gate:
            raise ValueError(f'Only matching results can be merged, got gates {common_gate} and {result.gate}')
        if common_options is None:
            common_options = result.options
        elif common_options != result.options:
            raise ValueError(f'Only matching results can be merged, got options {common_options} and {result.options}')
        if not all_parameters.keys().isdisjoint(result.parameters):
            raise ValueError('Only results with disjoint parameters sets can be merged')
        all_parameters.update(result.parameters)
    if common_gate is None or common_options is None:
        return None
    return PhasedFSimCalibrationResult(all_parameters, common_gate, common_options)