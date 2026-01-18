import threading
import time
from collections import Counter
import numpy
import pytest
from thinc.api import (
from thinc.compat import has_cupy_gpu
from ..util import make_tempdir
def test_nested_operator_contexts():
    m1 = create_model(name='a')
    m2 = create_model(name='b')
    assert Model._context_operators.get() == {}
    with Model.define_operators({'+': lambda a, b: a.name + b.name}):
        value = m1 + m2
        with pytest.raises(TypeError):
            value = m1 * m2
        with Model.define_operators({'*': lambda a, b: a.name + b.name}):
            with pytest.raises(TypeError):
                value = m1 + m2
            value = m1 * m2
            with Model.define_operators({'-': lambda a, b: a.name + b.name}):
                with pytest.raises(TypeError):
                    value = m1 + m2
                value = m1 - m2
            with pytest.raises(TypeError):
                value = m1 + m2
            value = m1 * m2
        value = m1 + m2
        with pytest.raises(TypeError):
            value = m1 * m2
    assert value == 'ab'
    assert Model._context_operators.get() == {}