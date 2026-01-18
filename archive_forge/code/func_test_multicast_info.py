import pickle
import types
import random
import pytest
from netaddr import (
def test_multicast_info():
    ip = IPAddress('224.0.1.173')
    assert ip.info.IPv4[0].designation == 'Multicast'
    assert ip.info.IPv4[0].prefix == '224/8'
    assert ip.info.IPv4[0].status == 'Reserved'
    assert ip.info.Multicast[0].address == '224.0.1.173'