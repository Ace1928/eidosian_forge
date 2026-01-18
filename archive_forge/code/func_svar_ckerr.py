import numpy as np
import numpy.linalg as npl
from numpy.linalg import slogdet
from statsmodels.tools.decorators import deprecated_alias
from statsmodels.tools.numdiff import approx_fprime, approx_hess
import statsmodels.tsa.base.tsa_model as tsbase
from statsmodels.tsa.vector_ar.irf import IRAnalysis
import statsmodels.tsa.vector_ar.util as util
from statsmodels.tsa.vector_ar.var_model import VARProcess, VARResults
def svar_ckerr(svar_type, A, B):
    if A is None and (svar_type == 'A' or svar_type == 'AB'):
        raise ValueError('SVAR of type A or AB but A array not given.')
    if B is None and (svar_type == 'B' or svar_type == 'AB'):
        raise ValueError('SVAR of type B or AB but B array not given.')