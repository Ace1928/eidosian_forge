from __future__ import annotations
import os
import re
import subprocess
from monty.tempfile import ScratchDir
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.core import Element, Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
@staticmethod
def get_relaxed_structure(gout: str):
    """
        Args:
            gout (str): GULP output string.

        Returns:
            Structure: relaxed structure.
        """
    structure_lines = []
    cell_param_lines = []
    output_lines = gout.split('\n')
    n_lines = len(output_lines)
    idx = 0
    a = b = c = alpha = beta = gamma = 0.0
    while idx < n_lines:
        line = output_lines[idx]
        if 'Full cell parameters' in line:
            idx += 2
            line = output_lines[idx]
            a = float(line.split()[8])
            alpha = float(line.split()[11])
            line = output_lines[idx + 1]
            b = float(line.split()[8])
            beta = float(line.split()[11])
            line = output_lines[idx + 2]
            c = float(line.split()[8])
            gamma = float(line.split()[11])
            idx += 3
            break
        if 'Cell parameters' in line:
            idx += 2
            line = output_lines[idx]
            a = float(line.split()[2])
            alpha = float(line.split()[5])
            line = output_lines[idx + 1]
            b = float(line.split()[2])
            beta = float(line.split()[5])
            line = output_lines[idx + 2]
            c = float(line.split()[2])
            gamma = float(line.split()[5])
            idx += 3
            break
        idx += 1
    while idx < n_lines:
        line = output_lines[idx]
        if 'Final fractional coordinates of atoms' in line:
            idx += 6
            line = output_lines[idx]
            while line[0:2] != '--':
                structure_lines.append(line)
                idx += 1
                line = output_lines[idx]
            idx += 9
            line = output_lines[idx]
            if 'Final cell parameters' in line:
                idx += 3
                for del_i in range(6):
                    line = output_lines[idx + del_i]
                    cell_param_lines.append(line)
            break
        idx += 1
    if structure_lines:
        sp = []
        coords = []
        for line in structure_lines:
            fields = line.split()
            if fields[2] == 'c':
                sp.append(fields[1])
                coords.append([float(x) for x in fields[3:6]])
    else:
        raise OSError('No structure found')
    if cell_param_lines:
        a = float(cell_param_lines[0].split()[1])
        b = float(cell_param_lines[1].split()[1])
        c = float(cell_param_lines[2].split()[1])
        alpha = float(cell_param_lines[3].split()[1])
        beta = float(cell_param_lines[4].split()[1])
        gamma = float(cell_param_lines[5].split()[1])
    if not all([a, b, c, alpha, beta, gamma]):
        raise ValueError(f'Missing lattice parameters in Gulp output: a={a!r}, b={b!r}, c={c!r}, alpha={alpha!r}, beta={beta!r}, gamma={gamma!r}')
    lattice = Lattice.from_parameters(a, b, c, alpha, beta, gamma)
    return Structure(lattice, sp, coords)