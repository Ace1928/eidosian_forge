from __future__ import annotations
import re
from io import StringIO
from typing import TYPE_CHECKING, cast
import pandas as pd
from monty.io import zopen
from pymatgen.core import Molecule, Structure
from pymatgen.core.structure import SiteCollection
def as_dataframe(self):
    """
        Generates a coordinates data frame with columns: atom, x, y, and z
        In case of multiple frame XYZ, returns the last frame.

        Returns:
            pandas.DataFrame
        """
    lines = str(self)
    sio = StringIO(lines)
    df_xyz = pd.read_csv(sio, header=None, skiprows=(0, 1), comment='#', delim_whitespace=True, names=('atom', 'x', 'y', 'z'))
    df_xyz.index += 1
    return df_xyz