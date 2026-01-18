import os
from copy import deepcopy
from nibabel import load
import numpy as np
from ... import logging
from ...utils import spm_docs as sd
from ..base import (
from ..base.traits_extension import NoDefaultSpecified
from ..matlab import MatlabCommand
from ...external.due import due, Doi, BibTeX
def no_spm():
    """Checks if SPM is NOT installed
    used with pytest.mark.skipif decorator to skip tests
    that will fail if spm is not installed"""
    if 'NIPYPE_NO_MATLAB' in os.environ or Info.version() is None:
        return True
    else:
        return False