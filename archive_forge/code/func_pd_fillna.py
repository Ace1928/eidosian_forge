import numpy as np
import scipy
import scipy.sparse.linalg
import scipy.stats
import threadpoolctl
import sklearn
from ..externals._packaging.version import parse as parse_version
from .deprecation import deprecated
def pd_fillna(pd, frame):
    pd_version = parse_version(pd.__version__).base_version
    if parse_version(pd_version) < parse_version('2.2'):
        frame = frame.fillna(value=np.nan)
    else:
        with pd.option_context('future.no_silent_downcasting', True):
            frame = frame.fillna(value=np.nan).infer_objects(copy=False)
    return frame