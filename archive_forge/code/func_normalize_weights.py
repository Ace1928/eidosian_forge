import tensorflow.compat.v2 as tf
from keras.src.initializers import TruncatedNormal
from keras.src.layers.rnn import Wrapper
from tensorflow.python.util.tf_export import keras_export
def normalize_weights(self):
    """Generate spectral normalized weights.

        This method will update the value of `self.kernel` with the
        spectral normalized value, so that the layer is ready for `call()`.
        """
    weights = tf.reshape(self.kernel, [-1, self.kernel_shape[-1]])
    vector_u = self.vector_u
    if not tf.reduce_all(tf.equal(weights, 0.0)):
        for _ in range(self.power_iterations):
            vector_v = tf.math.l2_normalize(tf.matmul(vector_u, weights, transpose_b=True))
            vector_u = tf.math.l2_normalize(tf.matmul(vector_v, weights))
        vector_u = tf.stop_gradient(vector_u)
        vector_v = tf.stop_gradient(vector_v)
        sigma = tf.matmul(tf.matmul(vector_v, weights), vector_u, transpose_b=True)
        self.vector_u.assign(tf.cast(vector_u, self.vector_u.dtype))
        self.kernel.assign(tf.cast(tf.reshape(self.kernel / sigma, self.kernel_shape), self.kernel.dtype))