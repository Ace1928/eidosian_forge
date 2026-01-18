from warnings import warn
import numpy as np
import scipy.sparse as sparse
import osqp._osqp as _osqp
def linsys_solver_str_to_int(settings):
    linsys_solver_str = settings.pop('linsys_solver', '')
    if not isinstance(linsys_solver_str, str):
        raise TypeError('Setting linsys_solver ' + 'is required to be a string.')
    linsys_solver_str = linsys_solver_str.lower()
    if linsys_solver_str == 'qdldl':
        settings['linsys_solver'] = _osqp.constant('QDLDL_SOLVER')
    elif linsys_solver_str == 'mkl pardiso':
        settings['linsys_solver'] = _osqp.constant('MKL_PARDISO_SOLVER')
    elif linsys_solver_str == '':
        settings['linsys_solver'] = _osqp.constant('QDLDL_SOLVER')
    else:
        warn('Linear system solver not recognized. ' + 'Using default solver QDLDL.')
        settings['linsys_solver'] = _osqp.constant('QDLDL_SOLVER')
    return settings