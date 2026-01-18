from __future__ import annotations
import re
from collections import defaultdict
from typing import TYPE_CHECKING
import numpy as np
from monty.io import zopen
from monty.json import MSONable
from pymatgen.core import Element
from pymatgen.electronic_structure.core import Orbital, Spin
from pymatgen.electronic_structure.dos import CompleteDos, Dos
from pymatgen.io.feff import Header, Potential, Tags
@staticmethod
def charge_transfer_from_file(feff_inp_file, ldos_file):
    """
        Get charge transfer from file.

        Args:
            feff_inp_file (str): name of feff.inp file for run
            ldos_file (str): ldos filename for run, assume consecutive order,
                i.e., ldos01.dat, ldos02.dat....

        Returns:
            dictionary of dictionaries in order of potential sites
            ({"p": 0.154, "s": 0.078, "d": 0.0, "tot": 0.232}, ...)
        """
    cht = {}
    parameters = Tags.from_file(feff_inp_file)
    if 'RECIPROCAL' in parameters:
        dicts = [{}]
        pot_dict = {}
        dos_index = 1
        begin = 0
        pot_inp = re.sub('feff.inp', 'pot.inp', feff_inp_file)
        pot_readstart = re.compile('.*iz.*lmaxsc.*xnatph.*xion.*folp.*')
        pot_readend = re.compile('.*ExternalPot.*switch.*')
        with zopen(pot_inp, mode='r') as potfile:
            for line in potfile:
                if len(pot_readend.findall(line)) > 0:
                    break
                if begin == 1:
                    z_number = int(line.strip().split()[0])
                    ele_name = Element.from_Z(z_number).name
                    if len(pot_dict) == 0:
                        pot_dict[0] = ele_name
                    elif len(pot_dict) > 0:
                        pot_dict[max(pot_dict) + 1] = ele_name
                    begin += 1
                    continue
                if begin == 2:
                    z_number = int(line.strip().split()[0])
                    ele_name = Element.from_Z(z_number).name
                    dicts[0][ele_name] = dos_index
                    dos_index += 1
                    if len(pot_dict) == 0:
                        pot_dict[0] = ele_name
                    elif len(pot_dict) > 0:
                        pot_dict[max(pot_dict) + 1] = ele_name
                if len(pot_readstart.findall(line)) > 0:
                    begin = 1
    else:
        pot_string = Potential.pot_string_from_file(feff_inp_file)
        dicts = Potential.pot_dict_from_str(pot_string)
        pot_dict = dicts[1]
    for idx in range(len(dicts[0]) + 1):
        if len(str(idx)) == 1:
            with zopen(f'{ldos_file}0{idx}.dat', mode='rt') as file:
                lines = file.readlines()
                s = float(lines[3].split()[2])
                p = float(lines[4].split()[2])
                d = float(lines[5].split()[2])
                f1 = float(lines[6].split()[2])
                tot = float(lines[1].split()[4])
                cht[str(idx)] = {pot_dict[idx]: {'s': s, 'p': p, 'd': d, 'f': f1, 'tot': tot}}
        else:
            with zopen(f'{ldos_file}{idx}.dat', mode='rt') as file:
                lines = file.readlines()
                s = float(lines[3].split()[2])
                p = float(lines[4].split()[2])
                d = float(lines[5].split()[2])
                f1 = float(lines[6].split()[2])
                tot = float(lines[1].split()[4])
                cht[str(idx)] = {pot_dict[idx]: {'s': s, 'p': p, 'd': d, 'f': f1, 'tot': tot}}
    return cht