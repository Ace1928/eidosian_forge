import pickle
import random
from netaddr import (
def test_eui48_vs_eui64():
    eui48 = EUI('01-00-00-01-00-00')
    assert int(eui48) == 1099511693312
    eui64 = EUI('00-00-01-00-00-01-00-00')
    assert int(eui64) == 1099511693312
    assert eui48 != eui64