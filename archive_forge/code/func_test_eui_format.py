import pickle
import random
from netaddr import (
def test_eui_format():
    mac = EUI('00-1B-77-49-54-FD')
    assert mac.format() == '00-1B-77-49-54-FD'
    assert mac.format(mac_pgsql) == '001b77:4954fd'
    assert mac.format(mac_unix_expanded) == '00:1b:77:49:54:fd'