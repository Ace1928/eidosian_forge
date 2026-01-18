import torch
from torch._export.db.case import export_case, SupportLevel
def recreate_list(self):
    return [torch.zeros(3, 2), torch.zeros(3, 2)]