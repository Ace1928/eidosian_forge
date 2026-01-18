from collections import OrderedDict
import os
import torch
from torch.serialization import default_restore_location
from typing import Any, Dict, List
from parlai.core.agents import create_agent
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.core.script import ParlaiScript
def load_fairseq_checkpoint(self):
    """
        Load a checkpoint to CPU (with upgrading for backward compatibility).

        :return state:
            loaded fairseq state
        """
    paths: List[str] = self.opt['input']
    if len(paths) == 1:
        return self._load_single_fairseq_checkpoint(paths[0])
    pieces = [self._load_single_fairseq_checkpoint(p) for p in paths]
    output_sd = {'args': pieces[0]['args']}
    output_model = {}
    pieces = {k: [p['model'][k] for p in pieces] for k in pieces[0]['model'].keys()}
    for k, subpieces in pieces.items():
        if '.version' in k:
            continue
        elif '_float_tensor' in k:
            output_model[k] = subpieces[0]
        elif 'out_proj.weight' in k or 'fc2.weight' in k:
            output_model[k] = torch.cat(subpieces, dim=1)
        elif 'out_proj.bias' in k or 'fc2.bias' in k:
            output_model[k] = subpieces[0]
        elif '_proj' in k or 'fc1' in k:
            output_model[k] = torch.cat(subpieces, dim=0)
        elif '_norm' in k:
            output_model[k] = subpieces[0]
        elif 'embed_tokens' in k:
            output_model[k] = torch.cat(subpieces, dim=0)
        else:
            print(f'Could not handle {k}')
            __import__('ipdb').set_trace()
            print()
    output_sd['model'] = output_model
    return output_sd