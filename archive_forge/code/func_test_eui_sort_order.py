import pickle
import random
from netaddr import (
def test_eui_sort_order():
    eui_list = [EUI(0, 64), EUI(0), EUI(281474976710655, dialect=mac_unix), EUI(281474976710656)]
    random.shuffle(eui_list)
    eui_list.sort()
    assert [(str(eui), eui.version) for eui in eui_list] == [('00-00-00-00-00-00', 48), ('ff:ff:ff:ff:ff:ff', 48), ('00-00-00-00-00-00-00-00', 64), ('00-01-00-00-00-00-00-00', 64)]