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
def get_action_dist_cls(self, framework: str):
    """Get the action distribution class.

        The default behavior is to get the action distribution from the
        `Catalog._action_dist_class_fn`.

        You should override this to have RLlib build your custom action
        distribution instead of the default one. For example, if you don't want to
        use RLlib's default RLModules with their default models, but only want to
        change the distribution that Catalog returns.

        Args:
            framework: The framework to use. Either "torch" or "tf2".

        Returns:
            The action distribution.
        """
    assert hasattr(self, '_action_dist_class_fn'), 'You must define a `Catalog._action_dist_class_fn` attribute in your Catalog subclass or override the `Catalog.action_dist_class_fn` method. By default, an action_dist_class_fn is created in the __post_init__ method.'
    return self._action_dist_class_fn(framework=framework)