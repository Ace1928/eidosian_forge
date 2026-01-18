import logging
import pickle
import torch
from torch.utils.data.dataset import Dataset
@staticmethod
def create_inst(inst, seq_len):
    output = {'input_ids_0': torch.tensor(inst['input_ids_0'], dtype=torch.long)[:seq_len]}
    output['mask_0'] = (output['input_ids_0'] != 0).float()
    if 'input_ids_1' in inst:
        output['input_ids_1'] = torch.tensor(inst['input_ids_1'], dtype=torch.long)[:seq_len]
        output['mask_1'] = (output['input_ids_1'] != 0).float()
    output['label'] = torch.tensor(inst['label'], dtype=torch.long)
    return output