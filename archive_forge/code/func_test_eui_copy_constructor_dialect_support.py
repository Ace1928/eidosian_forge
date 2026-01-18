import pickle
import random
from netaddr import (
def test_eui_copy_constructor_dialect_support():
    mac = EUI('00-1B-77-49-54-FD')
    copy = EUI(mac, dialect=mac_unix_expanded)
    assert str(copy) == '00:1b:77:49:54:fd'