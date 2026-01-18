from typing import Optional
from ray.rllib.algorithms.dreamerv3.tf.models.components.mlp import MLP
from ray.rllib.algorithms.dreamerv3.tf.models.components.reward_predictor_layer import (
from ray.rllib.algorithms.dreamerv3.utils import (
from ray.rllib.utils.framework import try_import_tf
def init_ema(self) -> None:
    """Initializes the EMA-copy of the critic from the critic's weights.

        After calling this method, the two networks have identical weights.
        """
    vars = self.mlp.trainable_variables + self.return_layer.trainable_variables
    vars_ema = self.mlp_ema.variables + self.return_layer_ema.variables
    assert len(vars) == len(vars_ema) and len(vars) > 0
    for var, var_ema in zip(vars, vars_ema):
        assert var is not var_ema
        var_ema.assign(var)