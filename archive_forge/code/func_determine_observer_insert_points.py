from typing import Any, Dict, Set, Tuple, Callable, List
import torch
import torch.nn as nn
import torch.ao.nn.qat as nnqat
from abc import ABC, abstractmethod
from torch.ao.quantization.fake_quantize import FakeQuantize
from torch.ao.quantization.fx.graph_module import GraphModule
from torch.ao.quantization.fx._model_report.model_report_observer import ModelReportObserver
from torch.ao.quantization.qconfig import (
from torch.ao.quantization.observer import (
from torch.ao.quantization.fx._equalize import (
from torch.ao.quantization.observer import _is_activation_post_process
def determine_observer_insert_points(self, prepared_fx_model: GraphModule) -> Dict[str, Dict[str, Any]]:
    """ Determines where observers need to be inserted for the Outlier Detector.

        For this detector, we want to place observers in front of supported layers.

        Currently inserts observers for:
            all layers that do not have children (leaf level layers)

        Args:
            prepared_fx_model (GraphModule):  The prepared Fx GraphModule

        Returns a Dict mapping from unique observer fqns (where we want to insert them) to a Dict with:
            key "target_node" -> the node we are trying to observe with this observer (torch.fx.node.Node)
            key "observer_to_insert" -> the observer we wish to insert (ObserverBase)
            key "is_post_observer" -> True if this is meant to be a post-observer for target_node, False if pre-observer
            key "observer_args" -> The arguments that are meant to be passed into the observer
        """
    obs_ctr = ModelReportObserver
    obs_fqn_to_info: Dict[str, Dict[str, Any]] = {}
    for fqn, module in prepared_fx_model.named_modules():
        if self._supports_insertion(module):
            targeted_node = self._get_targeting_node(prepared_fx_model, fqn)
            pre_obs_fqn = fqn + '.' + self.DEFAULT_PRE_OBSERVER_NAME
            obs_fqn_to_info[pre_obs_fqn] = {DETECTOR_TARGET_NODE_KEY: targeted_node, DETECTOR_OBS_TO_INSERT_KEY: obs_ctr(ch_axis=self.ch_axis, comp_percentile=self.reference_percentile), DETECTOR_IS_POST_OBS_KEY: False, DETECTOR_OBS_ARGS_KEY: targeted_node.args}
    return obs_fqn_to_info