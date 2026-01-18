import pickle
import random
from netaddr import (
def test_eui64_custom_dialect():

    class eui64_custom(eui64_unix):
        word_fmt = '%.2X'
    mac = EUI('00-1B-77-49-54-FD-12-34', dialect=eui64_custom)
    assert str(mac) == '00:1B:77:49:54:FD:12:34'