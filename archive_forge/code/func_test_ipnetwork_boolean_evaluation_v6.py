import pickle
import pytest
from netaddr import IPAddress, IPNetwork
def test_ipnetwork_boolean_evaluation_v6():
    assert bool(IPNetwork('::/0'))