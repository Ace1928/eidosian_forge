import six
import numpy as np
from patsy import PatsyError
from patsy.design_info import DesignMatrix, DesignInfo
from patsy.eval import EvalEnvironment
from patsy.desc import ModelDesc
from patsy.build import (design_matrix_builders,
from patsy.util import (have_pandas, asarray_or_pandas,
def incr_dbuilders(formula_like, data_iter_maker, eval_env=0, NA_action='drop'):
    """Construct two design matrix builders incrementally from a large data
    set.

    :func:`incr_dbuilders` is to :func:`incr_dbuilder` as :func:`dmatrices` is
    to :func:`dmatrix`. See :func:`incr_dbuilder` for details.
    """
    eval_env = EvalEnvironment.capture(eval_env, reference=1)
    design_infos = _try_incr_builders(formula_like, data_iter_maker, eval_env, NA_action)
    if design_infos is None:
        raise PatsyError('bad formula-like object')
    if len(design_infos[0].column_names) == 0:
        raise PatsyError('model is missing required outcome variables')
    return design_infos