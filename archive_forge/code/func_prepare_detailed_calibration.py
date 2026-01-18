from typing import Any, Dict, Set, Tuple, Callable
from collections import OrderedDict
import torch
from torch.ao.quantization.fx._model_report.detector import (
from torch.ao.quantization.fx._model_report.model_report_visualizer import ModelReportVisualizer
from torch.ao.quantization.fx.graph_module import GraphModule
from torch.ao.quantization.observer import ObserverBase
from torch.ao.quantization.qconfig_mapping import QConfigMapping, QConfig
from torch.ao.quantization.fx._equalize import EqualizationQConfig
def prepare_detailed_calibration(self) -> GraphModule:
    """
        Takes in a graph model and inserts the following observers:
        - ModelReportObserver

        Each observer is inserted based on the desired_reports into the relevant locations

        Right now, each report in self._desired_detector_names has independent insertions
            However, if a module already has a Observer of the same type, the insertion will not occur
            This is because all of the same type of Observer collect same information, so redundant

        Returns the same GraphModule with the observers inserted
        """
    if self._prepared_flag:
        raise ValueError('Already ran preparing detailed callibration. Run the report generation next after callibration.')
    insert_observers_fqns: Dict[str, Any] = {}
    for detector in self._desired_report_detectors:
        obs_fqn_to_info = detector.determine_observer_insert_points(self._model)
        insert_observers_fqns.update(obs_fqn_to_info)
        self._detector_name_to_observer_fqns[detector.get_detector_name()] = set(obs_fqn_to_info.keys())
    for observer_fqn in insert_observers_fqns:
        target_node = insert_observers_fqns[observer_fqn][DETECTOR_TARGET_NODE_KEY]
        insert_obs = insert_observers_fqns[observer_fqn][DETECTOR_OBS_TO_INSERT_KEY]
        insert_post = insert_observers_fqns[observer_fqn][DETECTOR_IS_POST_OBS_KEY]
        observer_args = insert_observers_fqns[observer_fqn][DETECTOR_OBS_ARGS_KEY]
        self._insert_observer_around_module(observer_fqn, target_node, insert_obs, observer_args, insert_post)
    self._prepared_flag = True
    return self._model