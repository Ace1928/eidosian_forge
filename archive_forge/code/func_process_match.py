from .polynomial import Polynomial
from .ptolemyVarietyPrimeIdealGroebnerBasis import PtolemyVarietyPrimeIdealGroebnerBasis
from . import processFileBase
from . import utilities
import snappy
import re
import sys
import tempfile
import subprocess
import shutil
def process_match(i, comp, free_vars, witnesses_txt, genus_txt):
    if i != 0:
        if not comp[0] == ',':
            raise ValueError('Parsing decomposition, expected separating comma.')
        comp = comp[1:].strip()
    if genus_txt.strip():
        genus = int(genus_txt)
    else:
        genus = None
    witnesses_txts = processFileBase.find_section(witnesses_txt, 'WITNESS')
    witnesses = [_parse_ideal_groebner_basis(utilities.join_long_lines_deleting_whitespace(t).strip(), py_eval, manifold_thunk, free_vars, [], genus) for t in witnesses_txts]
    return _parse_ideal_groebner_basis(comp, py_eval, manifold_thunk, free_vars, witnesses, genus)