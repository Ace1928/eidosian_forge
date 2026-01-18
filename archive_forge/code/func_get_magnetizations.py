from __future__ import annotations
import logging
import multiprocessing
import os
import re
from tabulate import tabulate
from pymatgen.apps.borg.hive import SimpleVaspToComputedEntryDrone, VaspToComputedEntryDrone
from pymatgen.apps.borg.queen import BorgQueen
from pymatgen.io.vasp import Outcar
def get_magnetizations(dir: str, ion_list: list[int]):
    """Get magnetization info from OUTCARs.

    Args:
        dir (str): Directory name
        ion_list (list[int]): List of ions to obtain magnetization information for.

    Returns:
        int: 0 if successful.
    """
    data = []
    max_row = 0
    for parent, _subdirs, files in os.walk(dir):
        for file in files:
            if re.match('OUTCAR*', file):
                try:
                    row = []
                    fullpath = os.path.join(parent, file)
                    outcar = Outcar(fullpath)
                    mags = outcar.magnetization
                    mags = [m['tot'] for m in mags]
                    all_ions = list(range(len(mags)))
                    row.append(fullpath.lstrip('./'))
                    if ion_list:
                        all_ions = ion_list
                    for ion in all_ions:
                        row.append(str(mags[ion]))
                    data.append(row)
                    if len(all_ions) > max_row:
                        max_row = len(all_ions)
                except Exception:
                    pass
    for d in data:
        if len(d) < max_row + 1:
            d.extend([''] * (max_row + 1 - len(d)))
    headers = ['Filename']
    for i in range(max_row):
        headers.append(str(i))
    print(tabulate(data, headers))
    return 0