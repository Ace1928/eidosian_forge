import pickle
import random
from netaddr import (
def test_eui64():
    eui = EUI('00-1B-77-FF-FE-49-54-FD')
    assert eui == EUI('00-1B-77-FF-FE-49-54-FD')
    assert eui.oui == OUI('00-1B-77')
    assert eui.ei == 'FF-FE-49-54-FD'
    assert eui.eui64() == EUI('00-1B-77-FF-FE-49-54-FD')