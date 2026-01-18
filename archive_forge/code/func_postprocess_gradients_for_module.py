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
@OverrideToImplementCustomLogic_CallToSuperRecommended
def postprocess_gradients_for_module(self, *, module_id: ModuleID, hps: LearnerHyperparameters, module_gradients_dict: ParamDict) -> ParamDict:
    """Applies postprocessing operations on the gradients of the given module.

        Args:
            module_id: The module ID for which we will postprocess computed gradients.
                Note that `module_gradients_dict` already only carries those gradient
                tensors that belong to this `module_id`. Other `module_id`'s gradients
                are not available in this call.
            hps: The LearnerHyperparameters specific to the given `module_id`.
            module_gradients_dict: A dictionary of gradients in the same (flat) format
                as self._params, mapping gradient refs to gradient tensors, which are to
                be postprocessed. You may alter these tensors in place or create new
                ones and return these in a new dict.

        Returns:
            A dictionary with the updated gradients and the exact same (flat) structure
            as the incoming `module_gradients_dict` arg.
        """
    postprocessed_grads = {}
    if hps.grad_clip is None:
        postprocessed_grads.update(module_gradients_dict)
        return postprocessed_grads
    for optimizer_name, optimizer in self.get_optimizers_for_module(module_id):
        grad_dict_to_clip = self.filter_param_dict_for_optimizer(param_dict=module_gradients_dict, optimizer=optimizer)
        global_norm = self._get_clip_function()(grad_dict_to_clip, grad_clip=hps.grad_clip, grad_clip_by=hps.grad_clip_by)
        if hps.grad_clip_by == 'global_norm':
            self.register_metric(module_id, f'gradients_{optimizer_name}_global_norm', global_norm)
        postprocessed_grads.update(grad_dict_to_clip)
    return postprocessed_grads