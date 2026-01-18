import pickle
import random
from netaddr import (
def test_eui_pickle_support():
    eui1 = EUI('00-00-00-01-02-03')
    eui2 = pickle.loads(pickle.dumps(eui1))
    assert eui1 == eui2
    eui1 = EUI('00-00-00-01-02-03', dialect=mac_cisco)
    eui2 = pickle.loads(pickle.dumps(eui1))
    assert eui1 == eui2
    assert eui1.dialect == eui2.dialect
    oui1 = EUI('00-00-00-01-02-03').oui
    oui2 = pickle.loads(pickle.dumps(oui1))
    assert oui1 == oui2
    assert oui1.records == oui2.records
    iab1 = EUI('00-50-C2-00-1F-FF').iab
    iab2 = pickle.loads(pickle.dumps(iab1))
    assert iab1 == iab2
    assert iab1.record == iab2.record