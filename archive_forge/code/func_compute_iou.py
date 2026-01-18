import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
import torch.distributed as dist
from torch import IntTensor, Tensor
from torchmetrics.detection.helpers import _fix_empty_tensors, _input_validator
from torchmetrics.metric import Metric
from torchmetrics.utilities.data import _cumsum
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE, _PYCOCOTOOLS_AVAILABLE, _TORCHVISION_GREATER_EQUAL_0_8
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
def compute_iou(det: List[Any], gt: List[Any], iou_type: str='bbox') -> Tensor:
    """Compute IOU between detections and ground-truth using the specified iou_type."""
    from torchvision.ops import box_iou
    if iou_type == 'bbox':
        return box_iou(torch.stack(det), torch.stack(gt))
    if iou_type == 'segm':
        return _segm_iou(det, gt)
    raise Exception(f'IOU type {iou_type} is not supported')