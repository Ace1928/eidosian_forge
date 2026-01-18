import datetime
import json
import numpy as np
from ase.utils import reader, writer
def create_ase_object(objtype, dct):
    if objtype == 'cell':
        from ase.cell import Cell
        dct.pop('pbc', None)
        obj = Cell(**dct)
    elif objtype == 'bandstructure':
        from ase.spectrum.band_structure import BandStructure
        obj = BandStructure(**dct)
    elif objtype == 'bandpath':
        from ase.dft.kpoints import BandPath
        obj = BandPath(path=dct.pop('labelseq'), **dct)
    elif objtype == 'atoms':
        from ase import Atoms
        obj = Atoms.fromdict(dct)
    elif objtype == 'vibrationsdata':
        from ase.vibrations import VibrationsData
        obj = VibrationsData.fromdict(dct)
    else:
        raise ValueError('Do not know how to decode object type {} into an actual object'.format(objtype))
    assert obj.ase_objtype == objtype
    return obj