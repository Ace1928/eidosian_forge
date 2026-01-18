import abc
import dataclasses
import itertools
import os
import tempfile
import warnings
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, TYPE_CHECKING, Union
import numpy as np
import pandas as pd
import sympy
from cirq import circuits, study, ops, value, protocols
from cirq._doc import document
from cirq.work.observable_grouping import group_settings_greedy, GROUPER_T
from cirq.work.observable_measurement_data import (
from cirq.work.observable_settings import InitObsSetting, observables_to_settings, _MeasurementSpec
def measure_observables_df(circuit: 'cirq.AbstractCircuit', observables: Iterable['cirq.PauliString'], sampler: Union['cirq.Simulator', 'cirq.Sampler'], stopping_criteria: StoppingCriteria, *, readout_symmetrization: bool=False, circuit_sweep: Optional['cirq.Sweepable']=None, grouper: Union[str, GROUPER_T]=group_settings_greedy, readout_calibrations: Optional[BitstringAccumulator]=None, checkpoint: CheckpointFileOptions=CheckpointFileOptions()):
    """Measure observables and return resulting data as a Pandas dataframe.

    Please see `measure_observables` for argument documentation.
    """
    results = measure_observables(circuit=circuit, observables=observables, sampler=sampler, stopping_criteria=stopping_criteria, readout_symmetrization=readout_symmetrization, circuit_sweep=circuit_sweep, grouper=grouper, readout_calibrations=readout_calibrations, checkpoint=checkpoint)
    df = pd.DataFrame((res.as_dict() for res in results))
    return df