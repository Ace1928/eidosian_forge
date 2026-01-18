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
@tf_export('nn.silu', 'nn.swish')
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def swish(features, beta=1.0):
    """Computes the SiLU or Swish activation function: `x * sigmoid(beta * x)`.

  beta : Hyperparameter for Swish activation function. Default value 1.0.

  The SiLU activation function was introduced in "Gaussian Error Linear Units
  (GELUs)" [Hendrycks et al. 2016](https://arxiv.org/abs/1606.08415) and
  "Sigmoid-Weighted Linear Units for Neural Network Function Approximation in
  Reinforcement Learning"
  [Elfwing et al. 2017](https://arxiv.org/abs/1702.03118) and was independently
  discovered (and called swish) in "Searching for Activation Functions"
  [Ramachandran et al. 2017](https://arxiv.org/abs/1710.05941)

  Args:
    features: A `Tensor` representing preactivation values.
    beta: A 'Tensor' representing value of beta hyperparameter.

  Returns:
    The activation value.
  """
    features = ops.convert_to_tensor(features, name='features')
    beta = ops.convert_to_tensor(beta, name='beta')
    beta = math_ops.cast(beta, features.dtype)

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
    return swish_impl(features, beta)