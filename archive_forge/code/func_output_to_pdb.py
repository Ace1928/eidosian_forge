import math
import sys
from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.nn import LayerNorm
from ...integrations.deepspeed import is_deepspeed_available
from ...modeling_outputs import ModelOutput
from ...utils import (
from .configuration_esm import EsmConfig
from .modeling_esm import ESM_START_DOCSTRING, EsmModel, EsmPreTrainedModel
from .openfold_utils import (
@staticmethod
def output_to_pdb(output: Dict) -> List[str]:
    """Returns the pbd (file) string from the model given the model output."""
    output = {k: v.to('cpu').numpy() for k, v in output.items()}
    pdbs = []
    final_atom_positions = atom14_to_atom37(output['positions'][-1], output)
    final_atom_mask = output['atom37_atom_exists']
    for i in range(output['aatype'].shape[0]):
        aa = output['aatype'][i]
        pred_pos = final_atom_positions[i]
        mask = final_atom_mask[i]
        resid = output['residue_index'][i] + 1
        pred = OFProtein(aatype=aa, atom_positions=pred_pos, atom_mask=mask, residue_index=resid, b_factors=output['plddt'][i])
        pdbs.append(to_pdb(pred))
    return pdbs