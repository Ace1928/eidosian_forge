import abc
import json
import logging
import pathlib
from collections import defaultdict
from enum import Enum
from dataclasses import dataclass, field
from typing import (
import ray
from ray.rllib.core.learner.reduce_result_dict_fn import _reduce_mean_results
from ray.rllib.core.learner.scaling_config import LearnerGroupScalingConfig
from ray.rllib.core.rl_module.marl_module import (
from ray.rllib.core.rl_module.rl_module import (
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID, MultiAgentBatch
from ray.rllib.utils.annotations import (
from ray.rllib.utils.debug import update_global_seed_if_necessary
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.metrics import (
from ray.rllib.utils.minibatch_utils import (
from ray.rllib.utils.nested_dict import NestedDict
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.schedules.scheduler import Scheduler
from ray.rllib.utils.serialization import serialize_type
from ray.rllib.utils.typing import (
from ray.util.annotations import PublicAPI
@OverrideToImplementCustomLogic
def postprocess_gradients(self, gradients_dict: ParamDict) -> ParamDict:
    """Applies potential postprocessing operations on the gradients.

        This method is called after gradients have been computed and modifies them
        before they are applied to the respective module(s) by the optimizer(s).
        This might include grad clipping by value, norm, or global-norm, or other
        algorithm specific gradient postprocessing steps.

        This default implementation calls `self.postprocess_gradients_for_module()`
        on each of the sub-modules in our MultiAgentRLModule: `self.module` and
        returns the accumulated gradients dicts.

        Args:
            gradients_dict: A dictionary of gradients in the same (flat) format as
                self._params. Note that top-level structures, such as module IDs,
                will not be present anymore in this dict. It will merely map gradient
                tensor references to gradient tensors.

        Returns:
            A dictionary with the updated gradients and the exact same (flat) structure
            as the incoming `gradients_dict` arg.
        """
    postprocessed_gradients = {}
    for module_id in self.module.keys():
        module_grads_dict = {}
        for optimizer_name, optimizer in self.get_optimizers_for_module(module_id):
            module_grads_dict.update(self.filter_param_dict_for_optimizer(gradients_dict, optimizer))
        module_grads_dict = self.postprocess_gradients_for_module(module_id=module_id, hps=self.hps.get_hps_for_module(module_id), module_gradients_dict=module_grads_dict)
        assert isinstance(module_grads_dict, dict)
        postprocessed_gradients.update(module_grads_dict)
    return postprocessed_gradients