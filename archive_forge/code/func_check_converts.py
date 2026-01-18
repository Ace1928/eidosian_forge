import os
from typing import cast
from unittest import mock
import numpy as np
import pandas as pd
import pytest
import sympy
from google.protobuf import text_format
import cirq
import cirq_google
from cirq.experiments.xeb_fitting import XEBPhasedFSimCharacterizationOptions
from cirq_google.api import v2
from cirq_google.calibration.phased_fsim import (
from cirq_google.serialization.arg_func_langs import arg_to_proto
def check_converts(gate: cirq.Gate):
    result = try_convert_syc_or_sqrt_iswap_to_fsim(gate)
    assert np.allclose(cirq.unitary(gate), cirq.unitary(result))