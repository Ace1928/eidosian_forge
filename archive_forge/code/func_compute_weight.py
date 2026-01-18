import torch
from torch.nn.functional import normalize
from typing import Any, Optional, TypeVar
from ..modules import Module
def compute_weight(self, module: Module, do_power_iteration: bool) -> torch.Tensor:
    weight = getattr(module, self.name + '_orig')
    u = getattr(module, self.name + '_u')
    v = getattr(module, self.name + '_v')
    weight_mat = self.reshape_weight_to_matrix(weight)
    if do_power_iteration:
        with torch.no_grad():
            for _ in range(self.n_power_iterations):
                v = normalize(torch.mv(weight_mat.t(), u), dim=0, eps=self.eps, out=v)
                u = normalize(torch.mv(weight_mat, v), dim=0, eps=self.eps, out=u)
            if self.n_power_iterations > 0:
                u = u.clone(memory_format=torch.contiguous_format)
                v = v.clone(memory_format=torch.contiguous_format)
    sigma = torch.dot(u, torch.mv(weight_mat, v))
    weight = weight / sigma
    return weight