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
def maybe_to_json(self, obj: Any):
    """Call `cirq.to_json with `value` according to the configuration options in this class.

        If `checkpoint=False`, nothing will happen. Otherwise, we will use `checkpoint_fn` and
        `checkpoint_other_fn` as the destination JSON file as described in the class docstring.
        """
    if not self.checkpoint:
        return
    assert self.checkpoint_fn is not None, 'mypy'
    assert self.checkpoint_other_fn is not None, 'mypy'
    if os.path.exists(self.checkpoint_fn):
        os.replace(self.checkpoint_fn, self.checkpoint_other_fn)
    protocols.to_json(obj, self.checkpoint_fn)