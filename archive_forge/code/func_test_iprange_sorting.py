from ast import literal_eval
import pickle
import sys
import sys
import pytest
from netaddr import (
def test_iprange_sorting():
    ranges = ((IPAddress('::'), IPAddress('::')), (IPAddress('0.0.0.0'), IPAddress('255.255.255.255')), (IPAddress('::'), IPAddress('::255.255.255.255')), (IPAddress('0.0.0.0'), IPAddress('0.0.0.0')))
    assert sorted(ranges) == [(IPAddress('0.0.0.0'), IPAddress('0.0.0.0')), (IPAddress('0.0.0.0'), IPAddress('255.255.255.255')), (IPAddress('::'), IPAddress('::')), (IPAddress('::'), IPAddress('::255.255.255.255'))]