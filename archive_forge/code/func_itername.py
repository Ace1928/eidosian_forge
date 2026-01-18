from copy import deepcopy
import re
import numpy as np
from ... import config
from ...interfaces.base import DynamicTraitedSpec
from ...utils.filemanip import loadpkl, savepkl
@property
def itername(self):
    """Get the name of the expanded iterable."""
    itername = self._id
    if self._hierarchy:
        itername = '%s.%s' % (self._hierarchy, self._id)
    return itername