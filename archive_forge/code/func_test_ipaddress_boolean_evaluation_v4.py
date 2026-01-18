import pickle
import types
import random
import pytest
from netaddr import (
def test_ipaddress_boolean_evaluation_v4():
    assert not bool(IPAddress('0.0.0.0'))
    assert bool(IPAddress('0.0.0.1'))
    assert bool(IPAddress('255.255.255.255'))