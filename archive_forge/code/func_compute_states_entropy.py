from gymnasium.spaces import Box, Discrete, Space
import numpy as np
from typing import List, Optional, Union
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override, PublicAPI
from ray.rllib.utils.exploration.exploration import Exploration
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.from_config import from_config
from ray.rllib.utils.tf_utils import get_placeholder
from ray.rllib.utils.typing import FromConfigSpec, ModelConfigDict, TensorType
@PublicAPI
def compute_states_entropy(obs_embeds: np.ndarray, embed_dim: int, k_nn: int) -> np.ndarray:
    """Compute states entropy using K nearest neighbour method.

    Args:
        obs_embeds: Observation latent representation using
            encoder model.
        embed_dim: Embedding vector dimension.
        k_nn: Number of nearest neighbour for K-NN estimation.

    Returns:
        Computed states entropy.
    """
    obs_embeds_ = np.reshape(obs_embeds, [-1, embed_dim])
    dist = np.linalg.norm(obs_embeds_[:, None, :] - obs_embeds_[None, :, :], axis=-1)
    return dist.argsort(axis=-1)[:, :k_nn][:, -1].astype(np.float32)