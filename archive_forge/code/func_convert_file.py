import collections
import numpy
import os
import torch
from safetensors.torch import serialize_file, load_file
import argparse
def convert_file(pt_filename: str, sf_filename: str, rename={}, transpose_names=[]):
    loaded: collections.OrderedDict = torch.load(pt_filename, map_location='cpu')
    if 'state_dict' in loaded:
        loaded = loaded['state_dict']
    kk = list(loaded.keys())
    version = 4
    for x in kk:
        if 'ln_x' in x:
            version = max(5, version)
        if 'gate.weight' in x:
            version = max(5.1, version)
        if int(version) == 5 and 'att.time_decay' in x:
            if len(loaded[x].shape) > 1:
                if loaded[x].shape[1] > 1:
                    version = max(5.2, version)
        if 'time_maa' in x:
            version = max(6, version)
    print(f'Model detected: v{version:.1f}')
    if version == 5.1:
        _, n_emb = loaded['emb.weight'].shape
        for k in kk:
            if 'time_decay' in k or 'time_faaaa' in k:
                loaded[k] = loaded[k].unsqueeze(1).repeat(1, n_emb // loaded[k].shape[0])
    with torch.no_grad():
        for k in kk:
            new_k = rename_key(rename, k).lower()
            v = loaded[k].half()
            del loaded[k]
            for transpose_name in transpose_names:
                if transpose_name in new_k:
                    dims = len(v.shape)
                    v = v.transpose(dims - 2, dims - 1)
            print(f'{new_k}\t{v.shape}\t{v.dtype}')
            loaded[new_k] = {'dtype': str(v.dtype).split('.')[-1], 'shape': v.shape, 'data': v.numpy().tobytes()}
    dirname = os.path.dirname(sf_filename)
    os.makedirs(dirname, exist_ok=True)
    serialize_file(loaded, sf_filename, metadata={'format': 'pt'})