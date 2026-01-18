import numpy as np
import pytest
from opt_einsum import (backends, contract, contract_expression, helpers, sharing)
from opt_einsum.contract import Shaped, infer_backend, parse_backend
def test_auto_backend_custom_array_no_tensordot():
    x = Shaped((1, 2, 3))
    assert infer_backend(x) == 'opt_einsum'
    assert parse_backend([x], 'auto') == 'numpy'