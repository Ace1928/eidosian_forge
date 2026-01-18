import pytest
import srsly
from thinc.api import (
def test_multi_model_roundtrip_bytes():
    model = chain(Maxout(5, 10, nP=2), Maxout(2, 3)).initialize()
    b = model.layers[0].get_param('b')
    b += 1
    b = model.layers[1].get_param('b')
    b += 2
    data = model.to_bytes()
    b = model.layers[0].get_param('b')
    b -= 1
    b = model.layers[1].get_param('b')
    b -= 2
    model = model.from_bytes(data)
    assert model.layers[0].get_param('b')[0, 0] == 1
    assert model.layers[1].get_param('b')[0, 0] == 2