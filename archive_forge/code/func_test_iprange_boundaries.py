from ast import literal_eval
import pickle
import sys
import sys
import pytest
from netaddr import (
def test_iprange_boundaries():
    assert list(iter_iprange('192.0.2.0', '192.0.2.7')) == [IPAddress('192.0.2.0'), IPAddress('192.0.2.1'), IPAddress('192.0.2.2'), IPAddress('192.0.2.3'), IPAddress('192.0.2.4'), IPAddress('192.0.2.5'), IPAddress('192.0.2.6'), IPAddress('192.0.2.7')]
    assert list(iter_iprange('::ffff:192.0.2.0', '::ffff:192.0.2.7')) == [IPAddress('::ffff:192.0.2.0'), IPAddress('::ffff:192.0.2.1'), IPAddress('::ffff:192.0.2.2'), IPAddress('::ffff:192.0.2.3'), IPAddress('::ffff:192.0.2.4'), IPAddress('::ffff:192.0.2.5'), IPAddress('::ffff:192.0.2.6'), IPAddress('::ffff:192.0.2.7')]