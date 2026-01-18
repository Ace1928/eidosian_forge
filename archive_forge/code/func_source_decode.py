import numpy as np
from sklearn.base import BaseEstimator, is_classifier, is_regressor
from sklearn.gaussian_process.kernels import Kernel
import inspect
def source_decode(sourcecode, verbose=0):
    """Decode operator source and import operator class.

    Parameters
    ----------
    sourcecode: string
        a string of operator source (e.g 'sklearn.feature_selection.RFE')
    verbose: int, optional (default: 0)
        How much information TPOT communicates while it's running.
        0 = none, 1 = minimal, 2 = high, 3 = all.
        if verbose > 2 then ImportError will rasie during initialization


    Returns
    -------
    import_str: string
        a string of operator class source (e.g. 'sklearn.feature_selection')
    op_str: string
        a string of operator class (e.g. 'RFE')
    op_obj: object
        operator class (e.g. RFE)

    """
    tmp_path = sourcecode.split('.')
    op_str = tmp_path.pop()
    import_str = '.'.join(tmp_path)
    try:
        if sourcecode.startswith('tpot.'):
            exec('from {} import {}'.format(import_str[4:], op_str))
        else:
            exec('from {} import {}'.format(import_str, op_str))
        op_obj = eval(op_str)
    except Exception as e:
        if verbose > 2:
            raise ImportError('Error: could not import {}.\n{}'.format(sourcecode, e))
        else:
            print('Warning: {} is not available and will not be used by TPOT.'.format(sourcecode))
        op_obj = None
    return (import_str, op_str, op_obj)