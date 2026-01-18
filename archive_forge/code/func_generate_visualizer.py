from typing import Any, Dict, Set, Tuple, Callable
from collections import OrderedDict
import torch
from torch.ao.quantization.fx._model_report.detector import (
from torch.ao.quantization.fx._model_report.model_report_visualizer import ModelReportVisualizer
from torch.ao.quantization.fx.graph_module import GraphModule
from torch.ao.quantization.observer import ObserverBase
from torch.ao.quantization.qconfig_mapping import QConfigMapping, QConfig
from torch.ao.quantization.fx._equalize import EqualizationQConfig
def generate_visualizer(self) -> ModelReportVisualizer:
    """
        Generates a ModelReportVisualizer instance using the reports generated
        by the generate_model_report() method.

        Returns the generated ModelReportVisualizer instance initialized

        Note:
            Throws exception if attempt to get visualizers without generating report
        """
    if len(self._generated_reports) == 0:
        raise Exception('Unable to generate visualizers without first generating reports')
    module_fqns_to_features: OrderedDict = self._reformat_reports_for_visualizer()
    visualizer: ModelReportVisualizer = ModelReportVisualizer(module_fqns_to_features)
    return visualizer