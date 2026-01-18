import numpy as np
from numpy.testing import assert_, assert_allclose, assert_raises
import statsmodels.datasets.macrodata.data as macro
from statsmodels.tsa.vector_ar.tests.JMulTi_results.parse_jmulti_vecm_output import (
from statsmodels.tsa.vector_ar.var_model import VAR
from .JMulTi_results.parse_jmulti_var_output import (
def reorder_jmultis_det_terms(jmulti_output, constant, seasons):
    """
    In case of seasonal terms and a trend term we have to reorder them to make
    the outputs from JMulTi and statsmodels comparable.
    JMulTi's ordering is: [constant], [seasonal terms], [trend term] while
    in statsmodels it is: [constant], [trend term], [seasonal terms]

    Parameters
    ----------
    jmulti_output : ndarray (neqs x number_of_deterministic_terms)

    constant : bool
        Indicates whether there is a constant term or not in jmulti_output.
    seasons : int
        Number of seasons in the model. That means there are seasons-1
        columns for seasonal terms in jmulti_output

    Returns
    -------
    reordered : ndarray (neqs x number_of_deterministic_terms)
        jmulti_output reordered such that the order of deterministic terms
        matches that of statsmodels.
    """
    if seasons == 0:
        return jmulti_output
    constant = int(constant)
    const_column = jmulti_output[:, :constant]
    season_columns = jmulti_output[:, constant:constant + seasons - 1].copy()
    trend_columns = jmulti_output[:, constant + seasons - 1:].copy()
    return np.hstack((const_column, trend_columns, season_columns))