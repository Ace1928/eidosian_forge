from typing import Sequence
import numpy as np
import pytest
import cirq
def test_make_experiment_no_rots():
    exp = cirq.experiments.StateTomographyExperiment([cirq.LineQubit(0), cirq.LineQubit(1), cirq.LineQubit(2)])
    assert len(exp.rot_sweep) > 0