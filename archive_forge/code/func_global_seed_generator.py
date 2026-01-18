import random as python_random
import numpy as np
from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.backend.common import global_state
from keras.src.utils import jax_utils
def global_seed_generator():
    if jax_utils.is_in_jax_tracing_scope():
        raise ValueError('[JAX RNG] When tracing a JAX function, you should only use seeded random ops, e.g. you should create a `SeedGenerator` instance, attach it to your layer/model, and pass the instance as the `seed` argument when calling random ops. Unseeded random ops would get incorrectly traced by JAX and would become constant after tracing. Example:\n\n```\n# Make sure to set the seed generator as a layer attribute\nself.seed_generator = keras.random.SeedGenerator(seed=1337)\n...\nout = keras.random.normal(shape=(1,), seed=self.seed_generator)\n```')
    gen = global_state.get_global_attribute('global_seed_generator')
    if gen is None:
        gen = SeedGenerator()
        global_state.set_global_attribute('global_seed_generator', gen)
    return gen