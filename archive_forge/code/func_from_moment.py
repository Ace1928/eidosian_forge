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
@classmethod
def from_moment(cls, moment: cirq.Moment, options: FloquetPhasedFSimCalibrationOptions):
    """Creates a FloquetPhasedFSimCalibrationRequest from a Moment.

        Given a `Moment` object, this function extracts out the pairs of
        qubits and the `Gate` used to create a `FloquetPhasedFSimCalibrationRequest`
        object.  The moment must contain only identical two-qubit FSimGates.
        If dissimilar gates are passed in, a ValueError is raised.
        """
    pairs, gate = _create_pairs_from_moment(moment)
    return cls(pairs, gate, options)