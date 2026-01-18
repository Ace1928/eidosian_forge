from __future__ import annotations
import itertools
import json
import logging
import math
import os
import platform
import re
import sys
import warnings
from enum import Enum, unique
from time import sleep
from typing import TYPE_CHECKING, Any, Literal
import requests
from monty.json import MontyDecoder, MontyEncoder
from ruamel.yaml import YAML
from tqdm import tqdm
from pymatgen.core import SETTINGS, Composition, Element, Structure
from pymatgen.core import __version__ as PMG_VERSION
from pymatgen.core.surface import get_symmetrically_equivalent_miller_indices
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
from pymatgen.entries.computed_entries import ComputedEntry, ComputedStructureEntry
from pymatgen.entries.exp_entries import ExpEntry
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.due import Doi, due
def submit_vasp_directory(self, rootdir, authors, projects=None, references='', remarks=None, master_data=None, master_history=None, created_at=None, ncpus=None):
    """Assimilates all vasp run directories beneath a particular
        directory using BorgQueen to obtain structures, and then submits thhem
        to the Materials Project as SNL files. VASP related meta data like
        initial structure and final energies are automatically incorporated.

        Note:
            As of now, this MP REST feature is open only to a select group of
            users. Opening up submissions to all users is being planned for the future.

        Args:
            rootdir (str): Rootdir to start assimilating VASP runs from.
            authors: *List* of {"name":'', "email":''} dicts,
                *list* of Strings as 'John Doe <johndoe@gmail.com>',
                or a single String with commas separating authors. The same
                list of authors should apply to all runs.
            projects ([str]): List of Strings ['Project A', 'Project B'].
                This applies to all structures.
            references (str): A String in BibTeX format. Again, this applies to
                all structures.
            remarks ([str]): List of Strings ['Remark A', 'Remark B']
            master_data (dict): A free form dict. Namespaced at the root
                level with an underscore, e.g. {"_materialsproject":<custom
                data>}. This data is added to all structures detected in the
                directory, in addition to other vasp data on a per structure
                basis.
            master_history: A master history to be added to all entries.
            created_at (datetime): A datetime object
            ncpus (int): Number of cpus to use in using BorgQueen to
                assimilate. Defaults to None, which means serial.
        """
    from pymatgen.apps.borg.hive import VaspToComputedEntryDrone
    from pymatgen.apps.borg.queen import BorgQueen
    drone = VaspToComputedEntryDrone(inc_structure=True, data=['filename', 'initial_structure'])
    queen = BorgQueen(drone, number_of_drones=ncpus)
    queen.parallel_assimilate(rootdir)
    structures = []
    metadata = []
    histories = []
    for e in queen.get_data():
        structures.append(e.structure)
        meta_dict = {'_vasp': {'parameters': e.parameters, 'final_energy': e.energy, 'final_energy_per_atom': e.energy_per_atom, 'initial_structure': e.data['initial_structure'].as_dict()}}
        if 'history' in e.parameters:
            histories.append(e.parameters['history'])
        if master_data is not None:
            meta_dict.update(master_data)
        metadata.append(meta_dict)
    if master_history is not None:
        histories = master_history * len(structures)
    return self.submit_structures(structures, authors, projects=projects, references=references, remarks=remarks, data=metadata, histories=histories, created_at=created_at)