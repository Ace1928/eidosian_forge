from ast import literal_eval
import pickle
import sys
import sys
import pytest
from netaddr import (
def test_more_iprange_sorting():
    ipranges = (IPRange('192.0.2.40', '192.0.2.50'), IPRange('192.0.2.20', '192.0.2.30'), IPRange('192.0.2.1', '192.0.2.254'))
    assert sorted(ipranges) == [IPRange('192.0.2.1', '192.0.2.254'), IPRange('192.0.2.20', '192.0.2.30'), IPRange('192.0.2.40', '192.0.2.50')]
    ipranges = list(ipranges)
    ipranges.append(IPRange('192.0.2.45', '192.0.2.49'))
    assert sorted(ipranges) == [IPRange('192.0.2.1', '192.0.2.254'), IPRange('192.0.2.20', '192.0.2.30'), IPRange('192.0.2.40', '192.0.2.50'), IPRange('192.0.2.45', '192.0.2.49')]