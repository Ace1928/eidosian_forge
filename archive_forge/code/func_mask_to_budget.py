import warnings
from typing import Any, List, Optional
import torch
from torch import nn
from peft.tuners.lora import LoraLayer
from peft.tuners.tuners_utils import check_adapters_to_merge
from peft.utils import transpose
def mask_to_budget(self, model, budget):
    value_ipt = {}
    vector_ipt = {}
    triplet_ipt = {}
    for n, p in model.named_parameters():
        if f'lora_A.{self.adapter_name}' in n:
            entry_ipt = self._element_score(n)
            comb_ipt = torch.mean(entry_ipt, dim=1, keepdim=True)
            name_m = n.replace('lora_A', '%s')
            if name_m not in vector_ipt:
                vector_ipt[name_m] = [comb_ipt]
            else:
                vector_ipt[name_m].append(comb_ipt)
        if f'lora_B.{self.adapter_name}' in n:
            entry_ipt = self._element_score(n)
            comb_ipt = torch.mean(entry_ipt, dim=0, keepdim=False).view(-1, 1)
            name_m = n.replace('lora_B', '%s')
            if name_m not in vector_ipt:
                vector_ipt[name_m] = [comb_ipt]
            else:
                vector_ipt[name_m].append(comb_ipt)
        if f'lora_E.{self.adapter_name}' in n:
            entry_ipt = self._element_score(n)
            name_m = n.replace('lora_E', '%s')
            value_ipt[name_m] = entry_ipt
    all_score = []
    for name_m in vector_ipt:
        ipt_E = value_ipt[name_m]
        ipt_AB = torch.cat(vector_ipt[name_m], dim=1)
        sum_ipt = self._combine_ipt(ipt_E, ipt_AB)
        name_E = name_m % 'lora_E'
        triplet_ipt[name_E] = sum_ipt.view(-1, 1)
        all_score.append(sum_ipt.view(-1))
    mask_threshold = torch.kthvalue(torch.cat(all_score), k=self.init_bgt - budget)[0].item()
    rank_pattern = {}
    with torch.no_grad():
        for n, p in model.named_parameters():
            if f'lora_E.{self.adapter_name}' in n:
                p.masked_fill_(triplet_ipt[n] <= mask_threshold, 0.0)
                rank_pattern[n] = (~(triplet_ipt[n] <= mask_threshold)).view(-1).tolist()
    return rank_pattern