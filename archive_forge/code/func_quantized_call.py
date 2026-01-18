import re
import string
import numpy as np
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
def quantized_call(self, inputs):

    @ops.custom_gradient
    def einsum_with_inputs_gradient(inputs, kernel, kernel_scale):

        def grad_fn(*args, upstream=None):
            if upstream is None:
                upstream, = args
            _kernel_scale = kernel_scale
            if self._kernel_squeeze_axes:
                _kernel_scale = ops.expand_dims(_kernel_scale, axis=self._kernel_squeeze_axes)
            if self._kernel_expand_axes:
                _kernel_scale = ops.squeeze(_kernel_scale, axis=self._kernel_expand_axes)
            _kernel_scale = ops.transpose(_kernel_scale, self._kernel_reverse_transpose_axes)
            float_kernel = ops.divide(ops.cast(kernel, dtype=self.compute_dtype), _kernel_scale)
            inputs_grad = ops.einsum(self._custom_gradient_equation, upstream, float_kernel)
            return (inputs_grad, None, None)
        inputs, inputs_scale = self.inputs_quantizer(inputs)
        x = ops.einsum(self.equation, inputs, kernel)
        inputs_scale = ops.transpose(inputs_scale, self._input_transpose_axes)
        if self._input_expand_axes:
            inputs_scale = ops.expand_dims(inputs_scale, axis=self._input_expand_axes)
        if self._input_squeeze_axes:
            inputs_scale = ops.squeeze(inputs_scale, axis=self._input_squeeze_axes)
        x = ops.cast(x, self.compute_dtype)
        x = ops.divide(x, ops.multiply(inputs_scale, kernel_scale))
        return (x, grad_fn)
    x = einsum_with_inputs_gradient(inputs, ops.convert_to_tensor(self._kernel), ops.convert_to_tensor(self.kernel_scale))
    if self.lora_enabled:
        lora_x = ops.einsum(self.equation, inputs, self.lora_kernel_a)
        lora_x = ops.matmul(lora_x, self.lora_kernel_b)
        x = ops.add(x, lora_x)
    if self.bias is not None:
        x += self.bias
    if self.activation is not None:
        x = self.activation(x)
    return x