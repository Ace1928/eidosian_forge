import pickle
import types
import random
import pytest
from netaddr import (
def test_ipnetwork_bad_string_constructor():
    with pytest.raises(AddrFormatError):
        IPNetwork('foo')