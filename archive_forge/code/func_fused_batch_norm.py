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
@tf_export(v1=['nn.fused_batch_norm'])
@dispatch.add_dispatch_support
def fused_batch_norm(x, scale, offset, mean=None, variance=None, epsilon=0.001, data_format='NHWC', is_training=True, name=None, exponential_avg_factor=1.0):
    """Batch normalization.


  See Source: [Batch Normalization: Accelerating Deep Network Training by
  Reducing Internal Covariate Shift; S. Ioffe, C. Szegedy]
  (http://arxiv.org/abs/1502.03167).

  Args:
    x: Input `Tensor` of 4 or 5 dimensions.
    scale: A `Tensor` of 1 dimension for scaling.
    offset: A `Tensor` of 1 dimension for bias.
    mean: A `Tensor` of 1 dimension for population mean. The shape and meaning
          of this argument depends on the value of is_training and
          exponential_avg_factor as follows:
          is_training==False (inference):
            Mean must be a `Tensor` of the same shape as scale containing the
            estimated population mean computed during training.
          is_training==True and exponential_avg_factor == 1.0:
            Mean must be None.
          is_training==True and exponential_avg_factor != 1.0:
            Mean must be a `Tensor` of the same shape as scale containing the
            exponential running mean.
    variance: A `Tensor` of 1 dimension for population variance. The shape and
          meaning of this argument depends on the value of is_training and
          exponential_avg_factor as follows:
          is_training==False (inference):
            Variance must be a `Tensor` of the same shape as scale containing
            the estimated population variance computed during training.
          is_training==True and exponential_avg_factor == 1.0:
            Variance must be None.
          is_training==True and exponential_avg_factor != 1.0:
            Variance must be a `Tensor` of the same shape as scale containing
            the exponential running variance.
    epsilon: A small float number added to the variance of x.
    data_format: The data format for x. Support "NHWC" (default) or "NCHW" for
                 4D tenors and "NDHWC" or "NCDHW" for 5D tensors.
    is_training: A bool value to specify if the operation is used for
                 training or inference.
    name: A name for this operation (optional).
    exponential_avg_factor: A float number (usually between 0 and 1) used
                            for controlling the decay of the running
                            population average of mean and variance.
                            If set to 1.0, the current batch average is
                            returned.

  Returns:
    y: A 4D or 5D Tensor for the normalized, scaled, offsetted x.
    running_mean: A 1D Tensor for the exponential running mean of x.
                  The output value is (1 - exponential_avg_factor) * mean +
                  exponential_avg_factor * batch_mean), where batch_mean
                  is the mean of the current batch in x.
    running_var: A 1D Tensor for the exponential running variance
                 The output value is (1 - exponential_avg_factor) * variance +
                 exponential_avg_factor * batch_variance), where batch_variance
                 is the variance of the current batch in x.

  References:
    Batch Normalization - Accelerating Deep Network Training by Reducing
    Internal Covariate Shift:
      [Ioffe et al., 2015](http://proceedings.mlr.press/v37/ioffe15.html)
      ([pdf](http://proceedings.mlr.press/v37/ioffe15.pdf))
  """
    if (not is_training or exponential_avg_factor != 1.0) and (mean is None or variance is None):
        raise ValueError(f'Both `mean` and `variance` must be a 1D tensor when `is_training` is False or `exponential_avg_factor` != 1.0. Received: `mean` {mean!r} and `variance` {variance!r}')
    x = ops.convert_to_tensor(x, name='input')
    scale = ops.convert_to_tensor(scale, name='scale')
    offset = ops.convert_to_tensor(offset, name='offset')
    if mean is None:
        mean = constant_op.constant([])
    if variance is None:
        variance = constant_op.constant([])
    y, running_mean, running_var, _, _, _ = gen_nn_ops.fused_batch_norm_v3(x, scale, offset, mean, variance, epsilon=epsilon, exponential_avg_factor=exponential_avg_factor, data_format=data_format, is_training=is_training, name=name)
    return (y, running_mean, running_var)