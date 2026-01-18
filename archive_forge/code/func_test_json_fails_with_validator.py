import numpy as np
import pytest
import cirq
def test_json_fails_with_validator():
    with pytest.raises(ValueError, match='not json serializable'):
        _ = cirq.to_json(cirq.LinearDict({}, validator=lambda: True))