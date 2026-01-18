import dataclasses
from typing import Optional, Tuple
import numpy as np
from onnx.reference.op_run import OpRun
def suppress_by_iou(boxes_data: np.ndarray, box_index1: int, box_index2: int, center_point_box: int, iou_threshold: float) -> bool:
    box1 = boxes_data[box_index1]
    box2 = boxes_data[box_index2]
    if center_point_box == 0:
        x1_min, x1_max = max_min(box1[1], box1[3])
        x2_min, x2_max = max_min(box2[1], box2[3])
        intersection_x_min = max(x1_min, x2_min)
        intersection_x_max = min(x1_max, x2_max)
        if intersection_x_max <= intersection_x_min:
            return False
        y1_min, y1_max = max_min(box1[0], box1[2])
        y2_min, y2_max = max_min(box2[0], box2[2])
        intersection_y_min = max(y1_min, y2_min)
        intersection_y_max = min(y1_max, y2_max)
        if intersection_y_max <= intersection_y_min:
            return False
    else:
        box1_width_half = box1[2] / 2
        box1_height_half = box1[3] / 2
        box2_width_half = box2[2] / 2
        box2_height_half = box2[3] / 2
        x1_min = box1[0] - box1_width_half
        x1_max = box1[0] + box1_width_half
        x2_min = box2[0] - box2_width_half
        x2_max = box2[0] + box2_width_half
        intersection_x_min = max(x1_min, x2_min)
        intersection_x_max = min(x1_max, x2_max)
        if intersection_x_max <= intersection_x_min:
            return False
        y1_min = box1[1] - box1_height_half
        y1_max = box1[1] + box1_height_half
        y2_min = box2[1] - box2_height_half
        y2_max = box2[1] + box2_height_half
        intersection_y_min = max(y1_min, y2_min)
        intersection_y_max = min(y1_max, y2_max)
        if intersection_y_max <= intersection_y_min:
            return False
    intersection_area = (intersection_x_max - intersection_x_min) * (intersection_y_max - intersection_y_min)
    if intersection_area <= 0:
        return False
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - intersection_area
    if area1 <= 0 or area2 <= 0 or union_area <= 0:
        return False
    intersection_over_union = intersection_area / union_area
    return intersection_over_union > iou_threshold