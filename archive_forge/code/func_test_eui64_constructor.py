import pickle
import random
from netaddr import (
def test_eui64_constructor():
    addr_colons = EUI('00:1B:77:49:54:FD:BB:34')
    assert addr_colons == EUI('00-1B-77-49-54-FD-BB-34')
    addr_no_delimiter = EUI('001B774954FDBB34')
    assert addr_no_delimiter == EUI('00-1B-77-49-54-FD-BB-34')