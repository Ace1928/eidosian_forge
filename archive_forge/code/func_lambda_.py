import random
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf
def lambda_(x):
    eager_out = tf.py_function(self.forward_eager, [x], tf.float32)
    with tf1.control_dependencies([eager_out]):
        eager_out.set_shape(x.shape)
        return eager_out