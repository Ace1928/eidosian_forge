import os
from enum import Enum, auto
from time import time
from typing import Any, Dict, List, Mapping, Optional, Union
import numpy as np
from numpy.random import default_rng
from cmdstanpy import _TMPDIR
from cmdstanpy.utils import (
def compose_command(self, idx: int, csv_file: str, *, diagnostic_file: Optional[str]=None, profile_file: Optional[str]=None) -> List[str]:
    """
        Compose CmdStan command for non-default arguments.
        """
    cmd: List[str] = []
    if idx is not None and self.chain_ids is not None:
        if idx < 0 or idx > len(self.chain_ids) - 1:
            raise ValueError('index ({}) exceeds number of chains ({})'.format(idx, len(self.chain_ids)))
        cmd.append(self.model_exe)
        cmd.append(f'id={self.chain_ids[idx]}')
    else:
        cmd.append(self.model_exe)
    if self.seed is not None:
        if not isinstance(self.seed, list):
            cmd.append('random')
            cmd.append(f'seed={self.seed}')
        else:
            cmd.append('random')
            cmd.append(f'seed={self.seed[idx]}')
    if self.data is not None:
        cmd.append('data')
        cmd.append(f'file={self.data}')
    if self.inits is not None:
        if not isinstance(self.inits, list):
            cmd.append(f'init={self.inits}')
        else:
            cmd.append(f'init={self.inits[idx]}')
    cmd.append('output')
    cmd.append(f'file={csv_file}')
    if diagnostic_file:
        cmd.append(f'diagnostic_file={diagnostic_file}')
    if profile_file:
        cmd.append(f'profile_file={profile_file}')
    if self.refresh is not None:
        cmd.append(f'refresh={self.refresh}')
    if self.sig_figs is not None:
        cmd.append(f'sig_figs={self.sig_figs}')
    cmd = self.method_args.compose(idx, cmd)
    return cmd