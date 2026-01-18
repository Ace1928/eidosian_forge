import functools
import torch
import torch._custom_ops
import torch.library
import torchvision.extension  # noqa: F401
@torch._custom_ops.impl_abstract('torchvision::nms')
def meta_nms(dets, scores, iou_threshold):
    torch._check(dets.dim() == 2, lambda: f'boxes should be a 2d tensor, got {dets.dim()}D')
    torch._check(dets.size(1) == 4, lambda: f'boxes should have 4 elements in dimension 1, got {dets.size(1)}')
    torch._check(scores.dim() == 1, lambda: f'scores should be a 1d tensor, got {scores.dim()}')
    torch._check(dets.size(0) == scores.size(0), lambda: f'boxes and scores should have same number of elements in dimension 0, got {dets.size(0)} and {scores.size(0)}')
    ctx = torch._custom_ops.get_ctx()
    num_to_keep = ctx.create_unbacked_symint()
    return dets.new_empty(num_to_keep, dtype=torch.long)