import os
import operator as op
import re
import warnings
from collections import OrderedDict
from os import path
import numpy as np
from ase.atoms import Atoms
from ase.calculators.singlepoint import (SinglePointDFTCalculator,
from ase.calculators.calculator import kpts2ndarray, kpts2sizeandoffsets
from ase.dft.kpoints import kpoint_convert
from ase.constraints import FixAtoms, FixCartesian
from ase.data import chemical_symbols, atomic_numbers
from ase.units import create_units
from ase.utils import iofunction
def read_fortran_namelist(fileobj):
    """Takes a fortran-namelist formatted file and returns nested
    dictionaries of sections and key-value data, followed by a list
    of lines of text that do not fit the specifications.

    Behaviour is taken from Quantum ESPRESSO 5.3. Parses fairly
    convoluted files the same way that QE should, but may not get
    all the MANDATORY rules and edge cases for very non-standard files:
        Ignores anything after '!' in a namelist, split pairs on ','
        to include multiple key=values on a line, read values on section
        start and end lines, section terminating character, '/', can appear
        anywhere on a line.
        All of these are ignored if the value is in 'quotes'.

    Parameters
    ----------
    fileobj : file
        An open file-like object.

    Returns
    -------
    data : dict of dict
        Dictionary for each section in the namelist with key = value
        pairs of data.
    card_lines : list of str
        Any lines not used to create the data, assumed to belong to 'cards'
        in the input file.

    """
    data = Namelist()
    card_lines = []
    in_namelist = False
    section = 'none'
    for line in fileobj:
        line = line.strip()
        if line.startswith('&'):
            section = line.split()[0][1:].lower()
            if section in data:
                section = '_ignored'
            data[section] = Namelist()
            in_namelist = True
        if not in_namelist and line:
            if line[0] not in ('!', '#'):
                card_lines.append(line)
        if in_namelist:
            key = []
            value = None
            in_quotes = False
            for character in line:
                if character == ',' and value is not None and (not in_quotes):
                    data[section][''.join(key).strip()] = str_to_value(''.join(value).strip())
                    key = []
                    value = None
                elif character == '=' and value is None and (not in_quotes):
                    value = []
                elif character == "'":
                    in_quotes = not in_quotes
                    value.append("'")
                elif character == '!' and (not in_quotes):
                    break
                elif character == '/' and (not in_quotes):
                    in_namelist = False
                    break
                elif value is not None:
                    value.append(character)
                else:
                    key.append(character)
            if value is not None:
                data[section][''.join(key).strip()] = str_to_value(''.join(value).strip())
    return (data, card_lines)