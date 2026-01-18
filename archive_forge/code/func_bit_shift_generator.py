from gymnasium.spaces import Box, Discrete
import numpy as np
from rllib.models.tf.attention_net import TrXLNet
from ray.rllib.utils.framework import try_import_tf
def bit_shift_generator(seq_length, shift, batch_size):
    while True:
        values = np.array([0.0, 1.0], dtype=np.float32)
        seq = np.random.choice(values, (batch_size, seq_length, 1))
        targets = np.squeeze(np.roll(seq, shift, axis=1).astype(np.int32))
        targets[:, :shift] = 0
        yield (seq, targets)