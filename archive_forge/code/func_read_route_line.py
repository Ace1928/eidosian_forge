from __future__ import annotations
import re
import warnings
from typing import TYPE_CHECKING
import numpy as np
import scipy.constants as cst
from monty.io import zopen
from scipy.stats import norm
from pymatgen.core import Composition, Element, Molecule
from pymatgen.core.operations import SymmOp
from pymatgen.core.units import Ha_to_eV
from pymatgen.electronic_structure.core import Spin
from pymatgen.util.coord import get_angle
from pymatgen.util.plotting import pretty_plot
def read_route_line(route):
    """
    read route line in gaussian input/output and return functional basis_set
    and a dictionary of other route parameters.

    Args:
        route (str) : the route line

    Returns:
        functional (str) : the method (HF, PBE ...)
        basis_set (str) : the basis set
        route (dict) : dictionary of parameters
    """
    scrf_patt = re.compile('^([sS][cC][rR][fF])\\s*=\\s*(.+)')
    multi_params_patt = re.compile('^([A-z]+[0-9]*)[\\s=]+\\((.*)\\)$')
    functional = basis_set = None
    route_params = {}
    dieze_tag = None
    if route:
        if '/' in route:
            tok = route.split('/')
            functional = tok[0].split()[-1]
            basis_set = tok[1].split()[0]
            for tok in [functional, basis_set, '/']:
                route = route.replace(tok, '')
        for tok in route.split():
            if (m := scrf_patt.match(tok)):
                route_params[m.group(1)] = m.group(2)
            elif tok.upper() in ['#', '#N', '#P', '#T']:
                dieze_tag = '#N' if tok == '#' else tok
                continue
            else:
                m = re.match(multi_params_patt, tok.strip('#'))
                if m:
                    pars = {}
                    for par in m.group(2).split(','):
                        p = par.split('=')
                        pars[p[0]] = None if len(p) == 1 else p[1]
                    route_params[m.group(1)] = pars
                else:
                    d = tok.strip('#').split('=')
                    route_params[d[0]] = None if len(d) == 1 else d[1]
    return (functional, basis_set, route_params, dieze_tag)