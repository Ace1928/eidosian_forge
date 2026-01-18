from dataclasses import dataclass
import numpy as np
import pytest
import cvxpy as cp
from cvxpy.atoms.quad_form import SymbolicQuadForm
from cvxpy.lin_ops.canon_backend import TensorRepresentation
from cvxpy.utilities.coeff_extractor import CoeffExtractor
@pytest.fixture
def coeff_extractor():
    inverset_data = MockeInverseData(var_offsets={1: 0}, x_length=2, var_shapes={1: (2,)}, param_shapes={2: (), 3: ()}, param_to_size={-1: 1, 2: 1, 3: 1}, param_id_map={2: 0, 3: 1, -1: 2})
    backend = cp.CPP_CANON_BACKEND
    return CoeffExtractor(inverset_data, backend)