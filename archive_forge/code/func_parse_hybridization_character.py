from __future__ import annotations
import copy
import logging
import math
import os
import re
import struct
import warnings
from typing import TYPE_CHECKING, Any
import networkx as nx
import numpy as np
import pandas as pd
from monty.io import zopen
from monty.json import MSONable, jsanitize
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import OpenBabelNN
from pymatgen.core import Molecule
from pymatgen.io.qchem.utils import (
def parse_hybridization_character(lines: list[str]) -> list[pd.DataFrame]:
    """
    Parse the hybridization character section of NBO output.

    Args:
        lines: QChem output lines.

    Returns:
        Data frames of formatted output.

    Raises:
        RuntimeError
    """
    orbitals = ['s', 'p', 'd', 'f']
    no_failures = True
    lp_and_bd_and_tc_dfs: list[pd.DataFrame] = []
    while no_failures:
        try:
            lines = jump_to_header(lines, '(Occupancy)   Bond orbital/ Coefficients/ Hybrids')
        except RuntimeError:
            try:
                lines = jump_to_header(lines, '(Occupancy)   Bond orbital / Coefficients / Hybrids')
            except RuntimeError:
                no_failures = False
        if no_failures:
            lines = lines[2:]
            lp_data = []
            bd_data = []
            tc_data = []
            i = -1
            while True:
                i += 1
                line = lines[i]
                if 'NHO DIRECTIONALITY AND BOND BENDING' in line:
                    break
                if 'Archival summary:' in line:
                    break
                if '3-Center, 4-Electron A:-B-:C Hyperbonds (A-B :C <=> A: B-C)' in line:
                    break
                if 'SECOND ORDER PERTURBATION THEORY ANALYSIS OF FOCK MATRIX IN NBO BASIS' in line:
                    break
                if 'Thank you very much for using Q-Chem.  Have a nice day.' in line:
                    break
                if 'LP' in line or 'LV' in line:
                    LPentry: dict[str, str | float] = dict.fromkeys(orbitals, 0.0)
                    LPentry['bond index'] = line[0:4].strip()
                    LPentry['occupancy'] = line[7:14].strip()
                    LPentry['type'] = line[16:19].strip()
                    LPentry['orbital index'] = line[20:22].strip()
                    LPentry['atom symbol'] = line[23:25].strip()
                    LPentry['atom number'] = line[25:28].strip()
                    for orbital in orbitals:
                        if orbital in line:
                            LPentry[orbital] = get_percentage(line, orbital)
                    i += 1
                    line = lines[i]
                    for orbital in orbitals:
                        if orbital in line:
                            LPentry[orbital] = get_percentage(line, orbital)
                    lp_data.append(LPentry)
                if 'BD' in line:
                    BDentry: dict[str, str | float] = {f'atom {i} {orbital}': 0.0 for orbital in orbitals for i in range(1, 3)}
                    BDentry['bond index'] = line[0:4].strip()
                    BDentry['occupancy'] = line[7:14].strip()
                    BDentry['type'] = line[16:19].strip()
                    BDentry['orbital index'] = line[20:22].strip()
                    BDentry['atom 1 symbol'] = line[23:25].strip()
                    BDentry['atom 1 number'] = line[25:28].strip()
                    BDentry['atom 2 symbol'] = line[29:31].strip()
                    BDentry['atom 2 number'] = line[31:34].strip()
                    i += 1
                    line = lines[i]
                    BDentry['atom 1 polarization'] = line[16:22].strip()
                    BDentry['atom 1 pol coeff'] = line[24:33].strip()
                    for orbital in orbitals:
                        if orbital in line:
                            BDentry[f'atom 1 {orbital}'] = get_percentage(line, orbital)
                    i += 1
                    line = lines[i]
                    for orbital in orbitals:
                        if orbital in line:
                            BDentry[f'atom 1 {orbital}'] = get_percentage(line, orbital)
                    while 's' not in line:
                        i += 1
                        line = lines[i]
                    BDentry['atom 2 polarization'] = line[16:22].strip()
                    BDentry['atom 2 pol coeff'] = line[24:33].strip()
                    for orbital in orbitals:
                        if orbital in line:
                            BDentry[f'atom 2 {orbital}'] = get_percentage(line, orbital)
                    i += 1
                    line = lines[i]
                    for orbital in orbitals:
                        if orbital in line:
                            BDentry[f'atom 2 {orbital}'] = get_percentage(line, orbital)
                    bd_data.append(BDentry)
                if '3C' in line:
                    TCentry: dict[str, str | float] = {f'atom {i} {orbital}': 0.0 for orbital in orbitals for i in range(1, 4)}
                    TCentry['bond index'] = line[0:4].strip()
                    TCentry['occupancy'] = line[7:14].strip()
                    TCentry['type'] = line[16:19].strip()
                    TCentry['orbital index'] = line[20:22].strip()
                    TCentry['atom 1 symbol'] = line[23:25].strip()
                    TCentry['atom 1 number'] = line[25:28].strip()
                    TCentry['atom 2 symbol'] = line[29:31].strip()
                    TCentry['atom 2 number'] = line[31:34].strip()
                    TCentry['atom 3 symbol'] = line[35:37].strip()
                    TCentry['atom 3 number'] = line[37:40].strip()
                    i += 1
                    line = lines[i]
                    TCentry['atom 1 polarization'] = line[16:22].strip()
                    TCentry['atom 1 pol coeff'] = line[24:33].strip()
                    for orbital in orbitals:
                        if orbital in line:
                            TCentry[f'atom 1 {orbital}'] = get_percentage(line, orbital)
                    i += 1
                    line = lines[i]
                    for orbital in orbitals:
                        if orbital in line:
                            TCentry[f'atom 1 {orbital}'] = get_percentage(line, orbital)
                    while 's' not in line:
                        i += 1
                        line = lines[i]
                    TCentry['atom 2 polarization'] = line[16:22].strip()
                    TCentry['atom 2 pol coeff'] = line[24:33].strip()
                    for orbital in orbitals:
                        if orbital in line:
                            TCentry[f'atom 2 {orbital}'] = get_percentage(line, orbital)
                    i += 1
                    line = lines[i]
                    for orbital in orbitals:
                        if orbital in line:
                            TCentry[f'atom 2 {orbital}'] = get_percentage(line, orbital)
                    while 's' not in line:
                        i += 1
                        line = lines[i]
                    TCentry['atom 3 polarization'] = line[16:22].strip()
                    TCentry['atom 3 pol coeff'] = line[24:33].strip()
                    for orbital in orbitals:
                        if orbital in line:
                            TCentry[f'atom 3 {orbital}'] = get_percentage(line, orbital)
                    i += 1
                    line = lines[i]
                    for orbital in orbitals:
                        if orbital in line:
                            TCentry[f'atom 3 {orbital}'] = get_percentage(line, orbital)
                    tc_data.append(TCentry)
            lp_and_bd_and_tc_dfs += (pd.DataFrame(lp_data), pd.DataFrame(bd_data), pd.DataFrame(tc_data))
    return lp_and_bd_and_tc_dfs