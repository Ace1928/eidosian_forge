import pickle
import random
from netaddr import (
def test_eui_dialects():
    mac = EUI('00-1B-77-49-54-FD')
    assert str(mac) == '00-1B-77-49-54-FD'
    mac = EUI('00-1B-77-49-54-FD', dialect=mac_unix)
    assert str(mac) == '0:1b:77:49:54:fd'
    mac = EUI('00-1B-77-49-54-FD', dialect=mac_unix_expanded)
    assert str(mac) == '00:1b:77:49:54:fd'
    mac = EUI('00-1B-77-49-54-FD', dialect=mac_cisco)
    assert str(mac) == '001b.7749.54fd'
    mac = EUI('00-1B-77-49-54-FD', dialect=mac_bare)
    assert str(mac) == '001B774954FD'
    mac = EUI('00-1B-77-49-54-FD', dialect=mac_pgsql)
    assert str(mac) == '001b77:4954fd'