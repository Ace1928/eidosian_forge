from pathlib import Path
from re import compile
import numpy as np
from ase import Atoms
from ase.utils import reader
from ase.units import Bohr
def read_fdf(fname):
    """Read a siesta style fdf-file.

    The data is returned as a dictionary
    ( label:value ).

    All labels are converted to lower case characters and
    are stripped of any '-', '_', or '.'.

    Ordinary values are stored as a list of strings (splitted on WS),
    and block values are stored as list of lists of strings
    (splitted per line, and on WS).
    If a label occurres more than once, the first occurrence
    takes precedence.

    The implementation applies no intelligence, and does not
    "understand" the data or the concept of units etc.
    Values are never parsed in any way, just stored as
    split strings.

    The implementation tries to comply with the fdf-format
    specification as presented in the siesta 2.0.2 manual.

    An fdf-dictionary could e.g. look like this::

        {'atomiccoordinatesandatomicspecies': [
              ['4.9999998', '5.7632392', '5.6095972', '1'],
              ['5.0000000', '6.5518100', '4.9929091', '2'],
              ['5.0000000', '4.9746683', '4.9929095', '2']],
         'atomiccoordinatesformat': ['Ang'],
         'chemicalspecieslabel': [['1', '8', 'O'],
                                  ['2', '1', 'H']],
         'dmmixingweight': ['0.1'],
         'dmnumberpulay': ['5'],
         'dmusesavedm': ['True'],
         'latticeconstant': ['1.000000', 'Ang'],
         'latticevectors': [
              ['10.00000000', '0.00000000', '0.00000000'],
              ['0.00000000', '11.52647800', '0.00000000'],
              ['0.00000000', '0.00000000', '10.59630900']],
         'maxscfiterations': ['120'],
         'meshcutoff': ['2721.139566', 'eV'],
         'numberofatoms': ['3'],
         'numberofspecies': ['2'],
         'paobasissize': ['dz'],
         'solutionmethod': ['diagon'],
         'systemlabel': ['H2O'],
         'wavefunckpoints': [['0.0', '0.0', '0.0']],
         'writedenchar': ['T'],
         'xcauthors': ['PBE'],
         'xcfunctional': ['GGA']}

    """
    fdf = {}
    lbz = _labelize
    lines = _read_fdf_lines(fname)
    while lines:
        w = lines.pop(0).split(None, 1)
        if lbz(w[0]) == '%block':
            if len(w) == 2:
                label = lbz(w[1])
                content = []
                while True:
                    if len(lines) == 0:
                        raise IOError('Unexpected EOF reached in %s, un-ended block %s' % (fname, label))
                    w = lines.pop(0).split()
                    if lbz(w[0]) == '%endblock':
                        break
                    content.append(w)
                if label not in fdf:
                    fdf[label] = content
            else:
                raise IOError('%%block statement without label')
        else:
            label = lbz(w[0])
            if len(w) == 1:
                fdf[label] = []
            else:
                fdf[label] = w[1].split()
    return fdf