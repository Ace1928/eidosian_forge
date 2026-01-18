import pickle
import random
from netaddr import (
def test_eui64_dialect_property_assignment():
    mac = EUI('00-1B-77-49-54-FD-12-34')
    assert str(mac) == '00-1B-77-49-54-FD-12-34'
    mac.dialect = eui64_cisco
    assert str(mac) == '001b.7749.54fd.1234'