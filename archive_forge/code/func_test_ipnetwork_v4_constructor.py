import pickle
import types
import random
import pytest
from netaddr import (
def test_ipnetwork_v4_constructor():
    assert IPNetwork('192.168.0.15') == IPNetwork('192.168.0.15/32')