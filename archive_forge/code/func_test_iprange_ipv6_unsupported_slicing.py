from ast import literal_eval
import pickle
import sys
import sys
import pytest
from netaddr import (
def test_iprange_ipv6_unsupported_slicing():
    with pytest.raises(TypeError):
        IPRange('::ffff:192.0.2.1', '::ffff:192.0.2.254')[0:10:2]