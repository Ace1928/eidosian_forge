import math
from enum import Enum, auto
from typing import Optional
import torch
from torch.autograd.profiler import record_function
from .base import FeatureMap
def pre_scale(self, x: torch.Tensor) -> torch.Tensor:
    with record_function('feature_map::pre_scale'):
        if self.iter_before_redraw is not None and self._iter_counter > self.iter_before_redraw or self.features is None or self.features.device != x.device:
            self._iter_counter = 1
            self.features = self._get_feature_map(x.shape[-1], self.dim_feature_map, x.device)
        features = self.features
        assert features is not None
        if features.dtype != x.dtype:
            self.features = features.to(x.dtype)
        self._iter_counter += 1
        if self.softmax_temp < 0:
            self.softmax_temp = x.shape[-1] ** (-0.25)
        x_scaled = x * self.softmax_temp
        norm_x_2 = torch.einsum('...d,...d->...', x_scaled, x_scaled).unsqueeze(-1)
        self.offset = -0.5 * norm_x_2 - self.h_scale + self.epsilon
        if self.normalize_inputs:
            self.offset -= norm_x_2.max(1, keepdim=True)[0]
    return x_scaled