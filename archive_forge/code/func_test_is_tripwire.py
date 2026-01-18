import pytest
from ..tripwire import TripWire, TripWireError, is_tripwire
def test_is_tripwire():
    assert not is_tripwire(object())
    assert is_tripwire(TripWire('some message'))