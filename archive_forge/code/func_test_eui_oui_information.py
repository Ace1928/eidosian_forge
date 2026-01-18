import pickle
import random
from netaddr import (
def test_eui_oui_information():
    mac = EUI('00-1B-77-49-54-FD')
    oui = mac.oui
    assert str(oui) == '00-1B-77'
    assert oui.registration().address == ['Lot 8, Jalan Hi-Tech 2/3', 'Kulim  Kedah  09000', 'MY']
    assert oui.registration().org == 'Intel Corporate'