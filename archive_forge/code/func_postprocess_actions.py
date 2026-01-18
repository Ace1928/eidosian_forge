import glob
import json
import logging
import math
import numpy as np
import os
from pathlib import Path
import random
import re
import tree  # pip install dm_tree
from typing import List, Optional, TYPE_CHECKING, Union
from urllib.parse import urlparse
import zipfile
from ray.rllib.offline.input_reader import InputReader
from ray.rllib.offline.io_context import IOContext
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import (
from ray.rllib.utils.annotations import override, PublicAPI, DeveloperAPI
from ray.rllib.utils.compression import unpack_if_needed
from ray.rllib.utils.spaces.space_utils import clip_action, normalize_action
from ray.rllib.utils.typing import Any, FileType, SampleBatchType
@DeveloperAPI
def postprocess_actions(batch: SampleBatchType, ioctx: IOContext) -> SampleBatchType:
    cfg = ioctx.config
    if cfg.get('clip_actions'):
        if ioctx.worker is None:
            raise ValueError('clip_actions is True but cannot clip actions since no workers exist')
        if isinstance(batch, SampleBatch):
            default_policy = ioctx.worker.policy_map.get(DEFAULT_POLICY_ID)
            batch[SampleBatch.ACTIONS] = clip_action(batch[SampleBatch.ACTIONS], default_policy.action_space_struct)
        else:
            for pid, b in batch.policy_batches.items():
                b[SampleBatch.ACTIONS] = clip_action(b[SampleBatch.ACTIONS], ioctx.worker.policy_map[pid].action_space_struct)
    if cfg.get('actions_in_input_normalized') is False and cfg.get('normalize_actions') is True:
        if ioctx.worker is None:
            raise ValueError('actions_in_input_normalized is False butcannot normalize actions since no workers exist')
        error_msg = 'Normalization of offline actions that are flattened is not supported! Make sure that you record actions into offline file with the `_disable_action_flattening=True` flag OR as already normalized (between -1.0 and 1.0) values. Also, when reading already normalized action values from offline files, make sure to set `actions_in_input_normalized=True` so that RLlib will not perform normalization on top.'
        if isinstance(batch, SampleBatch):
            pol = ioctx.worker.policy_map.get(DEFAULT_POLICY_ID)
            if isinstance(pol.action_space_struct, (tuple, dict)) and (not pol.config.get('_disable_action_flattening')):
                raise ValueError(error_msg)
            batch[SampleBatch.ACTIONS] = normalize_action(batch[SampleBatch.ACTIONS], pol.action_space_struct)
        else:
            for pid, b in batch.policy_batches.items():
                pol = ioctx.worker.policy_map[pid]
                if isinstance(pol.action_space_struct, (tuple, dict)) and (not pol.config.get('_disable_action_flattening')):
                    raise ValueError(error_msg)
                b[SampleBatch.ACTIONS] = normalize_action(b[SampleBatch.ACTIONS], ioctx.worker.policy_map[pid].action_space_struct)
    return batch