import collections
import numpy as np
import pandas as pd
import pytest
import cirq
import cirq.testing
from cirq.study.result import _pack_digits
def test_empty_measurements():
    assert cirq.ResultDict(params=None).repetitions == 0
    assert cirq.ResultDict(params=None, measurements={}).repetitions == 0
    assert cirq.ResultDict(params=None, records={}).repetitions == 0