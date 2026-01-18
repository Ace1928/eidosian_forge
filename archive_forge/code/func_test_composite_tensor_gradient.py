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
def test_composite_tensor_gradient(self):
    with self.session(graph=ops.Graph()) as sess:
        sess.graph.seed = random_seed.DEFAULT_GRAPH_SEED
        operator, _ = self.operator_and_matrix(shapes_info, dtype, use_placeholder=use_placeholder)
        x = self.make_x(operator, adjoint=False)
        y = operator.matmul(x)
        op_g, = gradients_impl.gradients(y, operator, grad_ys=array_ops.ones_like(y))

        def _unflatten_and_matmul(components):
            unflat_op = nest.pack_sequence_as(operator, components, expand_composites=True)
            return unflat_op.matmul(x)
        flat_op = nest.flatten(operator, expand_composites=True)
        y_ = _unflatten_and_matmul(flat_op)
        flat_g = gradients_impl.gradients(y_, flat_op, grad_ys=array_ops.ones_like(y_))
        if all((g is None for g in flat_g)):
            self.assertIsNone(op_g)
        else:
            self.assertIsInstance(op_g, operator.__class__)
            for g, ug in zip(nest.flatten(op_g, expand_composites=True), nest.flatten(flat_g, expand_composites=True)):
                self.assertAllClose(g, ug)