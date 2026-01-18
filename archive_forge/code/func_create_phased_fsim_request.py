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
def create_phased_fsim_request(self, pairs: Tuple[Tuple[cirq.Qid, cirq.Qid], ...], gate: cirq.Gate) -> 'FloquetPhasedFSimCalibrationRequest':
    return FloquetPhasedFSimCalibrationRequest(pairs=pairs, gate=gate, options=self)