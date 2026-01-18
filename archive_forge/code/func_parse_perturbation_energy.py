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
def parse_perturbation_energy(lines: list[str]) -> list[pd.DataFrame]:
    """
    Parse the perturbation energy section of NBO output.

    Args:
        lines: QChem output lines.

    Returns:
        Data frame of formatted output.

    Raises:
        RuntimeError
    """
    no_failures = True
    e2_dfs = []
    while no_failures:
        try:
            header_str = 'SECOND ORDER PERTURBATION THEORY ANALYSIS OF FOCK MATRIX IN NBO BASIS'
            lines = jump_to_header(lines, header_str)
        except RuntimeError:
            no_failures = False
        if no_failures:
            i = -1
            while True:
                i += 1
                line = lines[i]
                if 'within' in line:
                    lines = lines[i:]
                    break
            e2_data = []
            for line in lines:
                if 'NATURAL BOND ORBITALS' in line:
                    break
                if line.strip() == '':
                    continue
                if 'unit' in line:
                    continue
                if 'None' in line:
                    continue
                entry: dict[str, str | float] = {}
                first_3C = False
                second_3C = False
                if line[7] == '3':
                    first_3C = True
                if line[35] == '3':
                    second_3C = True
                if line[4] == '.':
                    entry['donor bond index'] = int(line[0:4].strip())
                    entry['donor type'] = str(line[5:9].strip())
                    entry['donor orbital index'] = int(line[10:12].strip())
                    entry['donor atom 1 symbol'] = str(line[13:15].strip())
                    entry['donor atom 1 number'] = int(line[15:17].strip())
                    entry['donor atom 2 symbol'] = str(line[18:20].strip())
                    entry['donor atom 2 number'] = z_int(line[20:22].strip())
                    entry['acceptor bond index'] = int(line[25:31].strip())
                    entry['acceptor type'] = str(line[32:36].strip())
                    entry['acceptor orbital index'] = int(line[37:39].strip())
                    entry['acceptor atom 1 symbol'] = str(line[40:42].strip())
                    entry['acceptor atom 1 number'] = int(line[42:44].strip())
                    entry['acceptor atom 2 symbol'] = str(line[45:47].strip())
                    entry['acceptor atom 2 number'] = z_int(line[47:49].strip())
                    entry['perturbation energy'] = float(line[50:62].strip())
                    entry['energy difference'] = float(line[62:70].strip())
                    entry['fock matrix element'] = float(line[70:79].strip())
                elif line[5] == '.':
                    entry['donor bond index'] = int(line[0:5].strip())
                    entry['donor type'] = str(line[6:10].strip())
                    entry['donor orbital index'] = int(line[11:13].strip())
                    if first_3C:
                        tmp = str(line[14:28].strip())
                        split = tmp.split('-')
                        if len(split) != 3:
                            raise ValueError('Should have three components! Exiting...')
                        entry['donor atom 1 symbol'] = split[0]
                        entry['donor atom 1 number'] = split[1]
                        entry['donor atom 2 symbol'] = split[2]
                        entry['donor atom 2 number'] = 'info_is_from_3C'
                    else:
                        entry['donor atom 1 symbol'] = str(line[14:16].strip())
                        entry['donor atom 1 number'] = int(line[16:19].strip())
                        entry['donor atom 2 symbol'] = str(line[20:22].strip())
                        entry['donor atom 2 number'] = z_int(line[22:25].strip())
                    entry['acceptor bond index'] = int(line[28:33].strip())
                    entry['acceptor type'] = str(line[34:38].strip())
                    entry['acceptor orbital index'] = int(line[39:41].strip())
                    if second_3C:
                        tmp = str(line[42:56].strip())
                        split = tmp.split('-')
                        if len(split) != 3:
                            raise ValueError('Should have three components! Exiting...')
                        entry['acceptor atom 1 symbol'] = split[0]
                        entry['acceptor atom 1 number'] = split[1]
                        entry['acceptor atom 2 symbol'] = split[2]
                        entry['acceptor atom 2 number'] = 'info_is_from_3C'
                    else:
                        entry['acceptor atom 1 symbol'] = str(line[42:44].strip())
                        entry['acceptor atom 1 number'] = int(line[44:47].strip())
                        entry['acceptor atom 2 symbol'] = str(line[48:50].strip())
                        entry['acceptor atom 2 number'] = z_int(line[50:53].strip())
                    try:
                        entry['perturbation energy'] = float(line[56:63].strip())
                    except ValueError:
                        if line[56:63].strip() == '*******':
                            entry['perturbation energy'] = float('inf')
                        else:
                            raise ValueError('Unknown value error in parsing perturbation energy')
                    entry['energy difference'] = float(line[63:71].strip())
                    entry['fock matrix element'] = float(line[71:79].strip())
                e2_data.append(entry)
            e2_dfs.append(pd.DataFrame(data=e2_data))
    return e2_dfs