import math
import numbers
import warnings
from typing import Any, List, Optional, Sequence, Tuple, Union
import PIL.Image
import torch
from torch.nn.functional import grid_sample, interpolate, pad as torch_pad
from torchvision import tv_tensors
from torchvision.transforms import _functional_pil as _FP
from torchvision.transforms._functional_tensor import _pad_symmetric
from torchvision.transforms.functional import (
from torchvision.utils import _log_api_usage_once
from ._meta import _get_size_image_pil, clamp_bounding_boxes, convert_bounding_box_format
from ._utils import _FillTypeJIT, _get_kernel, _register_five_ten_crop_kernel_internal, _register_kernel_internal
def elastic_bounding_boxes(bounding_boxes: torch.Tensor, format: tv_tensors.BoundingBoxFormat, canvas_size: Tuple[int, int], displacement: torch.Tensor) -> torch.Tensor:
    expected_shape = (1, canvas_size[0], canvas_size[1], 2)
    if not isinstance(displacement, torch.Tensor):
        raise TypeError('Argument displacement should be a Tensor')
    elif displacement.shape != expected_shape:
        raise ValueError(f'Argument displacement shape should be {expected_shape}, but given {displacement.shape}')
    if bounding_boxes.numel() == 0:
        return bounding_boxes
    device = bounding_boxes.device
    dtype = bounding_boxes.dtype if torch.is_floating_point(bounding_boxes) else torch.float32
    if displacement.dtype != dtype or displacement.device != device:
        displacement = displacement.to(dtype=dtype, device=device)
    original_shape = bounding_boxes.shape
    bounding_boxes = convert_bounding_box_format(bounding_boxes, old_format=format, new_format=tv_tensors.BoundingBoxFormat.XYXY).reshape(-1, 4)
    id_grid = _create_identity_grid(canvas_size, device=device, dtype=dtype)
    inv_grid = id_grid.sub_(displacement)
    points = bounding_boxes[:, [[0, 1], [2, 1], [2, 3], [0, 3]]].reshape(-1, 2)
    if points.is_floating_point():
        points = points.ceil_()
    index_xy = points.to(dtype=torch.long)
    index_x, index_y = (index_xy[:, 0], index_xy[:, 1])
    t_size = torch.tensor(canvas_size[::-1], device=displacement.device, dtype=displacement.dtype)
    transformed_points = inv_grid[0, index_y, index_x, :].add_(1).mul_(0.5 * t_size).sub_(0.5)
    transformed_points = transformed_points.reshape(-1, 4, 2)
    out_bbox_mins, out_bbox_maxs = torch.aminmax(transformed_points, dim=1)
    out_bboxes = clamp_bounding_boxes(torch.cat([out_bbox_mins, out_bbox_maxs], dim=1).to(bounding_boxes.dtype), format=tv_tensors.BoundingBoxFormat.XYXY, canvas_size=canvas_size)
    return convert_bounding_box_format(out_bboxes, old_format=tv_tensors.BoundingBoxFormat.XYXY, new_format=format, inplace=True).reshape(original_shape)