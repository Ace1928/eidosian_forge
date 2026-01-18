import numpy as np
import pytest
from ase.io.formats import ioformats, match_magic
@pytest.fixture
def lammpsdump():

    def factory(bounds='pp pp pp', position_cols='x y z', have_element=True, have_id=False, have_type=True):
        _element = 'element' if have_element else 'unk0'
        _id = 'id' if have_id else 'unk1'
        _type = 'type' if have_type else 'unk2'
        buf = f'        ITEM: TIMESTEP\n        0\n        ITEM: NUMBER OF ATOMS\n        3\n        ITEM: BOX BOUNDS {bounds}\n        0.0e+00 4e+00\n        0.0e+00 5.0e+00\n        0.0e+00 2.0e+01\n        ITEM: ATOMS {_element} {_id} {_type} {position_cols}\n        C  1 1 0.5 0.6 0.7\n        C  3 1 0.6 0.1 1.9\n        Si 2 2 0.45 0.32 0.67\n        '
        return buf
    return factory