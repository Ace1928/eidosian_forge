import tensorflow as tf
from tensorflow.experimental import numpy as tfnp
from keras.src.backend.common import standardize_dtype
from keras.src.backend.config import floatx
from keras.src.random.seed_generator import SeedGenerator
from keras.src.random.seed_generator import draw_seed
from keras.src.random.seed_generator import make_default_seed
def beta(shape, alpha, beta, dtype=None, seed=None):
    dtype = dtype or floatx()
    seed_1 = tf_draw_seed(seed)
    seed_2 = seed_1 + 12
    alpha = tf.convert_to_tensor(alpha, dtype=dtype)
    beta = tf.convert_to_tensor(beta, dtype=dtype)
    if tf.rank(alpha) > 1:
        alpha = tf.broadcast_to(alpha, shape)
    if tf.rank(beta) > 1:
        beta = tf.broadcast_to(beta, shape)
    gamma_a = tf.random.stateless_gamma(shape=shape, seed=seed_1, alpha=alpha, dtype=dtype)
    gamma_b = tf.random.stateless_gamma(shape=shape, seed=seed_2, alpha=beta, dtype=dtype)
    sample = gamma_a / (gamma_a + gamma_b)
    return sample