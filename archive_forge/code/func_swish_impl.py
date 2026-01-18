import math
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import candidate_sampling_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import custom_gradient
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import gen_array_ops  # pylint: disable=unused-import
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import gen_sparse_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.losses import util as losses_util
from tensorflow.python.platform import device_context
from tensorflow.python.util import dispatch
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.deprecation import deprecated_argument_lookup
from tensorflow.python.util.tf_export import tf_export
@custom_gradient.custom_gradient
def swish_impl(features, beta):

    def grad(dy):
        """Gradient for the Swish activation function."""
        with ops.control_dependencies([dy]):
            sigmoid_features = math_ops.sigmoid(beta * features)
        activation_grad = sigmoid_features * (1.0 + beta * features * (1.0 - sigmoid_features))
        beta_grad = math_ops.reduce_sum(dy * math_ops.square(features) * sigmoid_features * (1.0 - sigmoid_features))
        return (dy * activation_grad, beta_grad)
    return (features * math_ops.sigmoid(beta * features), grad)