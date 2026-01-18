from __future__ import annotations
import math
import os
import re
import textwrap
import warnings
from collections import defaultdict, deque
from functools import partial
from inspect import getfullargspec
from io import StringIO
from itertools import groupby
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
import numpy as np
from monty.dev import deprecated
from monty.io import zopen
from monty.serialization import loadfn
from pymatgen.core import Composition, DummySpecies, Element, Lattice, PeriodicSite, Species, Structure, get_el_sp
from pymatgen.core.operations import MagSymmOp, SymmOp
from pymatgen.electronic_structure.core import Magmom
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer, SpacegroupOperations
from pymatgen.symmetry.groups import SYMM_DATA, SpaceGroup
from pymatgen.symmetry.maggroups import MagneticSpaceGroup
from pymatgen.symmetry.structure import SymmetrizedStructure
from pymatgen.util.coord import find_in_coord_list_pbc, in_coord_list_pbc
def get_magsymops(self, data):
    """
        Equivalent to get_symops except for magnetic symmetry groups.
        Separate function since additional operation for time reversal symmetry
        (which changes magnetic moments on sites) needs to be returned.
        """
    mag_symm_ops = []
    bns_name = data.data.get('_space_group_magn.name_BNS')
    bns_num = data.data.get('_space_group_magn.number_BNS')
    if (xyzt := data.data.get('_space_group_symop_magn_operation.xyz')):
        if isinstance(xyzt, str):
            xyzt = [xyzt]
        mag_symm_ops = [MagSymmOp.from_xyzt_str(s) for s in xyzt]
        if data.data.get('_space_group_symop_magn_centering.xyz'):
            xyzt = data.data.get('_space_group_symop_magn_centering.xyz')
            if isinstance(xyzt, str):
                xyzt = [xyzt]
            centering_symops = [MagSymmOp.from_xyzt_str(s) for s in xyzt]
            all_ops = []
            for op in mag_symm_ops:
                for centering_op in centering_symops:
                    new_translation = [i - np.floor(i) for i in op.translation_vector + centering_op.translation_vector]
                    new_time_reversal = op.time_reversal * centering_op.time_reversal
                    all_ops.append(MagSymmOp.from_rotation_and_translation_and_time_reversal(rotation_matrix=op.rotation_matrix, translation_vec=new_translation, time_reversal=new_time_reversal))
            mag_symm_ops = all_ops
    elif bns_name or bns_num:
        label = bns_name or list(map(int, bns_num.split('.')))
        if data.data.get('_space_group_magn.transform_BNS_Pp_abc') != 'a,b,c;0,0,0':
            jonas_faithful = data.data.get('_space_group_magn.transform_BNS_Pp_abc')
            msg = MagneticSpaceGroup(label, jonas_faithful)
        elif data.data.get('_space_group_magn.transform_BNS_Pp'):
            return NotImplementedError('Incomplete specification to implement.')
        else:
            msg = MagneticSpaceGroup(label)
        mag_symm_ops = msg.symmetry_ops
    if not mag_symm_ops:
        msg = 'No magnetic symmetry detected, using primitive symmetry.'
        warnings.warn(msg)
        self.warnings.append(msg)
        mag_symm_ops = [MagSymmOp.from_xyzt_str('x, y, z, 1')]
    return mag_symm_ops