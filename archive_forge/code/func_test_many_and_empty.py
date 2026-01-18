import pytest
from preshed.maps import PreshMap
import random
def test_many_and_empty():
    table = PreshMap()
    for i in range(100, 110):
        table[i] = i
    for i in range(100, 110):
        del table[i]
    assert table[0] == None