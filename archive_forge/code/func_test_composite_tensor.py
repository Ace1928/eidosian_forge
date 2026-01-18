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
def test_composite_tensor(self):
    with self.session(graph=ops.Graph()) as sess:
        sess.graph.seed = random_seed.DEFAULT_GRAPH_SEED
        operator, mat = self.operator_and_matrix(shapes_info, dtype, use_placeholder=use_placeholder)
        self.assertIsInstance(operator, composite_tensor.CompositeTensor)
        flat = nest.flatten(operator, expand_composites=True)
        unflat = nest.pack_sequence_as(operator, flat, expand_composites=True)
        self.assertIsInstance(unflat, type(operator))
        x = self.make_x(operator, adjoint=False)
        op_y = def_function.function(lambda op: op.matmul(x))(unflat)
        mat_y = math_ops.matmul(mat, x)
        if not use_placeholder:
            self.assertAllEqual(mat_y.shape, op_y.shape)

        def body(op):
            return (type(op)(**op.parameters),)
        op_out, = while_v2.while_loop(cond=lambda _: True, body=body, loop_vars=(operator,), maximum_iterations=3)
        loop_y = op_out.matmul(x)
        op_y_, loop_y_, mat_y_ = sess.run([op_y, loop_y, mat_y])
        self.assertAC(op_y_, mat_y_)
        self.assertAC(loop_y_, mat_y_)
        nested_structure_coder.encode_structure(operator._type_spec)