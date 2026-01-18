import operator
import warnings
import numpy as np
from nibabel.optpkg import optional_package
from ..onetime import auto_attr as one_time
from ..openers import ImageOpener
from . import csareader as csar
from .dwiparams import B2q, nearest_pos_semi_def, q2bg
@one_time
def series_signature(self):
    """Add ICE dims from CSA header to signature"""
    signature = super().series_signature
    ice = csar.get_ice_dims(self.csa_header)
    if ice is not None:
        ice = ice[:6] + ice[8:9]
    signature['ICE_Dims'] = (ice, operator.eq)
    return signature