from keras.src import initializers
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.layers import Wrapper
from keras.src.layers.input_spec import InputSpec
from keras.src.utils.numerical_utils import normalize
def normalized_weights(self):
    """Generate spectral normalized weights.

        This method returns the updated value for `self.kernel` with the
        spectral normalized value, so that the layer is ready for `call()`.
        """
    weights = ops.reshape(self.kernel, [-1, self.kernel_shape[-1]])
    vector_u = self.vector_u.value
    for _ in range(self.power_iterations):
        vector_v = normalize(ops.matmul(vector_u, ops.transpose(weights)), axis=None)
        vector_u = normalize(ops.matmul(vector_v, weights), axis=None)
    sigma = ops.matmul(ops.matmul(vector_v, weights), ops.transpose(vector_u))
    kernel = ops.reshape(ops.divide(self.kernel, sigma), self.kernel_shape)
    return (ops.cast(vector_u, self.vector_u.dtype), ops.cast(kernel, self.kernel.dtype))