from collections.abc import (
import os
import posixpath
import numpy as np
from .._objects import phil, with_phil
from .. import h5d, h5i, h5r, h5p, h5f, h5t, h5s
from .compat import fspath, filename_encode
def default_lapl():
    """ Default link access property list """
    lapl = h5p.create(h5p.LINK_ACCESS)
    fapl = h5p.create(h5p.FILE_ACCESS)
    fapl.set_fclose_degree(h5f.CLOSE_STRONG)
    lapl.set_elink_fapl(fapl)
    return lapl