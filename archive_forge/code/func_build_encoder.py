import enum
import functools
from typing import Optional
import gymnasium as gym
import numpy as np
import tree
from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete, Tuple
from ray.rllib.core.models.base import Encoder
from ray.rllib.core.models.configs import (
from ray.rllib.core.models.configs import ModelConfig
from ray.rllib.models import MODEL_DEFAULTS
from ray.rllib.models.distributions import Distribution
from ray.rllib.models.preprocessors import get_preprocessor, Preprocessor
from ray.rllib.models.utils import get_filter_config
from ray.rllib.utils.deprecation import deprecation_warning
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.rllib.utils.spaces.simplex import Simplex
from ray.rllib.utils.spaces.space_utils import flatten_space
from ray.rllib.utils.spaces.space_utils import get_base_struct_from_space
from ray.rllib.utils.typing import ViewRequirementsDict
from ray.rllib.utils.annotations import (
@OverrideToImplementCustomLogic
def build_encoder(self, framework: str) -> Encoder:
    """Builds the encoder.

        By default, this method builds an encoder instance from Catalog._encoder_config.

        You should override this if you want to use RLlib's default RL Modules but
        only want to change the encoder. For example, if you want to use a custom
        encoder, but want to use RLlib's default heads, action distribution and how
        tensors are routed between them. If you want to have full control over the
        RL Module, we recommend writing your own RL Module by inheriting from one of
        RLlib's RL Modules instead.

        Args:
            framework: The framework to use. Either "torch" or "tf2".

        Returns:
            The encoder.
        """
    assert hasattr(self, '_encoder_config'), 'You must define a `Catalog._encoder_config` attribute in your Catalog subclass or override the `Catalog.build_encoder` method. By default, an encoder_config is created in the __post_init__ method.'
    return self._encoder_config.build(framework=framework)