from typing import Iterable, List, Sequence, Tuple
import numpy as np
import pytest
import cirq
def assert_not_implemented(val):
    with pytest.raises(TypeError, match='returned NotImplemented'):
        _ = cirq.kraus(val)
    assert cirq.kraus(val, None) is None
    assert cirq.kraus(val, NotImplemented) is NotImplemented
    assert cirq.kraus(val, (1,)) == (1,)
    assert cirq.kraus(val, LOCAL_DEFAULT) is LOCAL_DEFAULT
    assert not cirq.has_kraus(val)