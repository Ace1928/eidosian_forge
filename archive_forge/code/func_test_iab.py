import pickle
import random
from netaddr import (
def test_iab():
    eui = EUI('00-50-C2-05-C0-00')
    assert eui.is_iab()
    assert str(eui.oui) == '00-50-C2'
    assert str(eui.iab) == '00-50-C2-05-C0-00'
    assert eui.ei == '05-C0-00'
    assert int(eui.oui) == 20674
    assert int(eui.iab) == 84680796
    assert IAB(eui.value) == eui.iab