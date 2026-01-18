import numpy as np
from . import _translation_utils as translation_utils
from .... import symbol
def matrix_multiplication(attrs, inputs, proto_obj):
    """Performs general matrix multiplication"""
    return ('linalg_gemm2', attrs, inputs)