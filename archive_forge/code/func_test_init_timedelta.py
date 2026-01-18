from datetime import timedelta
import pytest
import sympy
import numpy as np
import cirq
from cirq.value import Duration
def test_init_timedelta():
    assert Duration(timedelta(microseconds=0)).total_picos() == 0
    assert Duration(timedelta(microseconds=513)).total_picos() == 513 * 10 ** 6
    assert Duration(timedelta(microseconds=-5)).total_picos() == -5 * 10 ** 6
    assert Duration(timedelta(microseconds=211)).total_picos() == 211 * 10 ** 6
    assert Duration(timedelta(seconds=3)).total_picos() == 3 * 10 ** 12
    assert Duration(timedelta(seconds=-5)).total_picos() == -5 * 10 ** 12
    assert Duration(timedelta(seconds=3)).total_nanos() == 3 * 10 ** 9
    assert Duration(timedelta(seconds=-5)).total_nanos() == -5 * 10 ** 9