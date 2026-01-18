import abc
import itertools
import numpy as np
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import test_util
from tensorflow.python.module import module
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import sort_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import while_v2
from tensorflow.python.ops.linalg import linalg_impl as linalg
from tensorflow.python.ops.linalg import linear_operator_util
from tensorflow.python.platform import test
from tensorflow.python.saved_model import load as load_model
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.saved_model import save as save_model
from tensorflow.python.util import nest
def test_adjoint(self):
    with self.test_session(graph=ops.Graph()) as sess:
        sess.graph.seed = random_seed.DEFAULT_GRAPH_SEED
        operator, mat = self.operator_and_matrix(shapes_info, dtype, use_placeholder=use_placeholder)
        op_adjoint = operator.adjoint().to_dense()
        op_adjoint_h = operator.H.to_dense()
        mat_adjoint = linalg.adjoint(mat)
        op_adjoint_v, op_adjoint_h_v, mat_adjoint_v = sess.run([op_adjoint, op_adjoint_h, mat_adjoint])
        self.assertAC(mat_adjoint_v, op_adjoint_v)
        self.assertAC(mat_adjoint_v, op_adjoint_h_v)