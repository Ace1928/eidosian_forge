from keras.src import activations
from keras.src import backend
from keras.src import constraints
from keras.src import dtype_policies
from keras.src import initializers
from keras.src import ops
from keras.src import quantizers
from keras.src import regularizers
from keras.src.api_export import keras_export
from keras.src.layers.input_spec import InputSpec
from keras.src.layers.layer import Layer
@ops.custom_gradient
def matmul_with_inputs_gradient(inputs, kernel, kernel_scale):

    def grad_fn(*args, upstream=None):
        if upstream is None:
            upstream, = args
        float_kernel = ops.divide(ops.cast(kernel, dtype=self.compute_dtype), kernel_scale)
        inputs_grad = ops.matmul(upstream, ops.transpose(float_kernel))
        return (inputs_grad, None, None)
    inputs, inputs_scale = self.inputs_quantizer(inputs)
    x = ops.matmul(inputs, kernel)
    x = ops.cast(x, self.compute_dtype)
    x = ops.divide(x, ops.multiply(inputs_scale, kernel_scale))
    return (x, grad_fn)