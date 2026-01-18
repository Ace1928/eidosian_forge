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
def test_gate_translators_are_consistent():

    def check(gate):
        result1 = try_convert_gate_to_fsim(gate)
        result2 = try_convert_sqrt_iswap_to_fsim(gate)
        assert result1 == result2
        assert result1 is not None
    check(cirq.FSimGate(theta=np.pi / 4, phi=0))
    check(cirq.FSimGate(theta=-np.pi / 4, phi=0))
    check(cirq.FSimGate(theta=7 * np.pi / 4, phi=0))
    check(cirq.PhasedFSimGate(theta=np.pi / 4, phi=0))
    check(cirq.ISwapPowGate(exponent=0.5))
    check(cirq.ISwapPowGate(exponent=-0.5))
    check(cirq.PhasedISwapPowGate(exponent=0.5, phase_exponent=-0.5))