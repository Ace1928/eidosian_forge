import numpy as np
import pickle
import ray
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.tf.recurrent_net import RecurrentNetwork
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf
@override(RecurrentNetwork)
def forward_rnn(self, inputs, state, seq_lens):
    RNNSpyModel.capture_index = 0
    model_out, value_out, h, c = self.base_model([inputs, seq_lens, state[0], state[1]])
    self._value_out = value_out
    return (model_out, [h, c])