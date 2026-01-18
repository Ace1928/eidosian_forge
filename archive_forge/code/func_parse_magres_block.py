import re
import numpy as np
from collections import OrderedDict
import ase.units
from ase.atoms import Atoms
from ase.spacegroup import Spacegroup
from ase.spacegroup.spacegroup import SpacegroupNotFoundError
from ase.calculators.singlepoint import SinglePointDFTCalculator
def parse_magres_block(block):
    """
            Parse magres block into data dictionary given list of record
            tuples.
        """
    name, records = block

    def ntensor33(name):
        return lambda d: {name: tensor33([float(x) for x in data])}

    def sitensor33(name):
        return lambda d: {'atom': {'label': data[0], 'index': int(data[1])}, name: tensor33([float(x) for x in data[2:]])}

    def sisitensor33(name):
        return lambda d: {'atom1': {'label': data[0], 'index': int(data[1])}, 'atom2': {'label': data[2], 'index': int(data[3])}, name: tensor33([float(x) for x in data[4:]])}
    tags = {'ms': sitensor33('sigma'), 'sus': ntensor33('S'), 'efg': sitensor33('V'), 'efg_local': sitensor33('V'), 'efg_nonlocal': sitensor33('V'), 'isc': sisitensor33('K'), 'isc_fc': sisitensor33('K'), 'isc_spin': sisitensor33('K'), 'isc_orbital_p': sisitensor33('K'), 'isc_orbital_d': sisitensor33('K'), 'units': check_units}
    data_dict = {}
    for record in records:
        tag, data = record
        if tag not in data_dict:
            data_dict[tag] = []
        data_dict[tag].append(tags[tag](data))
    return data_dict