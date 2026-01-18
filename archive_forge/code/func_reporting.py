import copy
import dataclasses
import gc
import logging
import tree  # pip install dm_tree
from typing import Any, Dict, List, Optional, Union
import gymnasium as gym
import numpy as np
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided
from ray.rllib.algorithms.dreamerv3.dreamerv3_catalog import DreamerV3Catalog
from ray.rllib.algorithms.dreamerv3.dreamerv3_learner import (
from ray.rllib.algorithms.dreamerv3.utils import do_symlog_obs
from ray.rllib.algorithms.dreamerv3.utils.env_runner import DreamerV3EnvRunner
from ray.rllib.algorithms.dreamerv3.utils.summaries import (
from ray.rllib.core.learner.learner import LearnerHyperparameters
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.models.catalog import MODEL_DEFAULTS
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID, SampleBatch
from ray.rllib.utils import deep_update
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.numpy import one_hot
from ray.rllib.utils.metrics import (
from ray.rllib.utils.replay_buffers.episode_replay_buffer import EpisodeReplayBuffer
from ray.rllib.utils.typing import LearningRateOrSchedule, ResultDict
@override(AlgorithmConfig)
def reporting(self, *, report_individual_batch_item_stats: Optional[bool]=NotProvided, report_dream_data: Optional[bool]=NotProvided, report_images_and_videos: Optional[bool]=NotProvided, **kwargs):
    """Sets the reporting related configuration.

        Args:
            report_individual_batch_item_stats: Whether to include loss and other stats
                per individual timestep inside the training batch in the result dict
                returned by `training_step()`. If True, besides the `CRITIC_L_total`,
                the individual critic loss values per batch row and time axis step
                in the train batch (CRITIC_L_total_B_T) will also be part of the
                results.
            report_dream_data:  Whether to include the dreamed trajectory data in the
                result dict returned by `training_step()`. If True, however, will
                slice each reported item in the dream data down to the shape.
                (H, B, t=0, ...), where H is the horizon and B is the batch size. The
                original time axis will only be represented by the first timestep
                to not make this data too large to handle.
            report_images_and_videos: Whether to include any image/video data in the
                result dict returned by `training_step()`.
            **kwargs:

        Returns:
            This updated AlgorithmConfig object.
        """
    super().reporting(**kwargs)
    if report_individual_batch_item_stats is not NotProvided:
        self.report_individual_batch_item_stats = report_individual_batch_item_stats
    if report_dream_data is not NotProvided:
        self.report_dream_data = report_dream_data
    if report_images_and_videos is not NotProvided:
        self.report_images_and_videos = report_images_and_videos
    return self