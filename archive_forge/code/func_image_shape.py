import operator
import warnings
import numpy as np
from nibabel.optpkg import optional_package
from ..onetime import auto_attr as one_time
from ..openers import ImageOpener
from . import csareader as csar
from .dwiparams import B2q, nearest_pos_semi_def, q2bg
@one_time
def image_shape(self):
    """Return image shape as returned by ``get_data()``"""
    rows = self.get('Rows')
    cols = self.get('Columns')
    if None in (rows, cols):
        return None
    return (rows // self.mosaic_size, cols // self.mosaic_size, self.n_mosaic)