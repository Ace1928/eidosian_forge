import pickle
import random
from netaddr import (
def test_eui_iab():
    mac = EUI('00-50-C2-00-0F-01')
    assert mac.is_iab()
    iab = mac.iab
    assert str(iab) == '00-50-C2-00-00-00'
    assert iab == IAB('00-50-C2-00-00-00')
    reg_info = iab.registration()
    assert reg_info.address == ['1241 Superieor Ave E', 'Cleveland  OH  44114', 'US']
    assert reg_info.iab == '00-50-C2-00-00-00'
    assert reg_info.org == 'T.L.S. Corp.'