from abc import abstractmethod, ABC
import functools
import warnings
import numpy as np
from typing import Dict, List
from ase.cell import Cell
from ase.build.bulk import bulk as newbulk
from ase.dft.kpoints import parse_path_string, sc_special_points, BandPath
from ase.utils import pbc2pbc
def bravaisclass(longname, crystal_family, lattice_system, pearson_symbol, parameters, variants, ndim=3):
    """Decorator for Bravais lattice classes.

    This sets a number of class variables and processes the information
    about different variants into a list of Variant objects."""

    def decorate(cls):
        btype = cls.__name__
        cls.name = btype
        cls.longname = longname
        cls.crystal_family = crystal_family
        cls.lattice_system = lattice_system
        cls.pearson_symbol = pearson_symbol
        cls.parameters = tuple(parameters)
        cls.variant_names = []
        cls.variants = {}
        cls.ndim = ndim
        for [name, special_point_names, special_path, special_points] in variants:
            cls.variant_names.append(name)
            cls.variants[name] = Variant(name, special_point_names, special_path, special_points)
        bravais_names.append(btype)
        bravais_lattices[btype] = cls
        bravais_classes[pearson_symbol] = cls
        return cls
    return decorate