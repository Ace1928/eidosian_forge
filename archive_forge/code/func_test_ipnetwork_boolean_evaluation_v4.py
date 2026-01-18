import pickle
import types
import random
import pytest
from netaddr import (
def test_ipnetwork_boolean_evaluation_v4():
    assert bool(IPNetwork('0.0.0.0/0'))