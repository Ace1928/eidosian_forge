import six
import numpy as np
from patsy import PatsyError
from patsy.design_info import DesignMatrix, DesignInfo
from patsy.eval import EvalEnvironment
from patsy.desc import ModelDesc
from patsy.build import (design_matrix_builders,
from patsy.util import (have_pandas, asarray_or_pandas,
def data_iter_maker():
    return iter([data])