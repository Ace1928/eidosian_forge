from typing import Any, Dict, Set, Tuple, Callable
from collections import OrderedDict
import torch
from torch.ao.quantization.fx._model_report.detector import (
from torch.ao.quantization.fx._model_report.model_report_visualizer import ModelReportVisualizer
from torch.ao.quantization.fx.graph_module import GraphModule
from torch.ao.quantization.observer import ObserverBase
from torch.ao.quantization.qconfig_mapping import QConfigMapping, QConfig
from torch.ao.quantization.fx._equalize import EqualizationQConfig
def generate_model_report(self, remove_inserted_observers: bool) -> Dict[str, Tuple[str, Dict]]:
    """
        Generates all the requested reports.

        Note:
            You should have callibrated the model with relevant data before calling this

        The reports generated are specified by the desired_reports specified in desired_reports

        Can optionally remove all the observers inserted by the ModelReport instance

        Args:
            remove_inserted_observers (bool): True to remove the observers inserted by this ModelReport instance

        Returns a mapping of each desired report name to a tuple with:
            The textual summary of that report information
            A dictionary containing relevant statistics or information for that report

        Note:
            Throws exception if we try to generate report on model we already removed observers from
            Throws exception if we try to generate report without preparing for callibration
        """
    if not self._prepared_flag:
        raise Exception('Cannot generate report without preparing model for callibration')
    if self._removed_observers:
        raise Exception('Cannot generate report on model you already removed observers from')
    reports_of_interest = {}
    for detector in self._desired_report_detectors:
        report_output = detector.generate_detector_report(self._model)
        reports_of_interest[detector.get_detector_name()] = report_output
    if remove_inserted_observers:
        self._removed_observers = True
        all_observers_of_interest: Set[str] = set()
        for desired_report in self._detector_name_to_observer_fqns:
            observers_of_interest = self._detector_name_to_observer_fqns[desired_report]
            all_observers_of_interest.update(observers_of_interest)
        for observer_fqn in all_observers_of_interest:
            self._model.delete_submodule(observer_fqn)
            node_obj = self._get_node_from_fqn(observer_fqn)
            if node_obj:
                self._model.graph.erase_node(node_obj)
            else:
                raise ValueError('Node no longer exists in GraphModule structure')
        self._model.recompile()
    saved_reports: Dict[str, Dict] = {report_name: report_tuple[1] for report_name, report_tuple in reports_of_interest.items()}
    self._generated_reports = saved_reports
    return reports_of_interest