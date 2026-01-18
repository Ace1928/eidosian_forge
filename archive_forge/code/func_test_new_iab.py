import pickle
import random
from netaddr import (
def test_new_iab():
    eui = EUI('40-D8-55-13-10-00')
    assert eui.is_iab()
    assert str(eui.oui) == '40-D8-55'
    assert str(eui.iab) == '40-D8-55-13-10-00'
    assert eui.ei == '13-10-00'
    assert int(eui.oui) == 4249685
    assert int(eui.iab) == 17406710065
    assert IAB(eui.value) == eui.iab