import pickle
import random
from netaddr import (
def test_mac_address_properties():
    mac = EUI('00-1B-77-49-54-FD')
    assert repr(mac) == "EUI('00-1B-77-49-54-FD')"
    assert str(mac)
    assert str(mac.oui) == '00-1B-77'
    assert mac.ei == '49-54-FD'
    assert mac.version == 48