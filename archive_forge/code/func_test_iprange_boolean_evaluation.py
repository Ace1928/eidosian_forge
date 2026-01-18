from ast import literal_eval
import pickle
import sys
import sys
import pytest
from netaddr import (
def test_iprange_boolean_evaluation():
    assert bool(IPRange('0.0.0.0', '255.255.255.255'))
    assert bool(IPRange('0.0.0.0', '0.0.0.0'))