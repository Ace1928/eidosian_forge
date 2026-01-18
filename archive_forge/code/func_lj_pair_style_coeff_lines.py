import pytest
import numpy as np
from ase import Atoms
def lj_pair_style_coeff_lines(lj_cutoff, eps):
    return [f'pair_style lj/cut {lj_cutoff}', f'pair_coeff * * {eps} 1']