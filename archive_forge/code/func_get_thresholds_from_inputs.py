import dataclasses
from typing import Optional, Tuple
import numpy as np
from onnx.reference.op_run import OpRun
def get_thresholds_from_inputs(self, pc: PrepareContext, max_output_boxes_per_class: int, iou_threshold: float, score_threshold: float) -> Tuple[int, float, float]:
    if pc.max_output_boxes_per_class_ is not None:
        max_output_boxes_per_class = max(pc.max_output_boxes_per_class_[0], 0)
    if pc.iou_threshold_ is not None:
        iou_threshold = pc.iou_threshold_[0]
    if pc.score_threshold_ is not None:
        score_threshold = pc.score_threshold_[0]
    return (max_output_boxes_per_class, iou_threshold, score_threshold)