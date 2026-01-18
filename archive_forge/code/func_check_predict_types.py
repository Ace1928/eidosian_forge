import os
import sys
from packaging.version import Version, parse
import numpy as np
from numpy.testing import assert_allclose, assert_
import pandas as pd
def check_predict_types(results):
    """
    Check that the `predict` method of the given results object produces the
    correct output type.

    Parameters
    ----------
    results : Results

    Raises
    ------
    AssertionError
    """
    res = results
    p_exog = np.squeeze(np.asarray(res.model.exog[:2]))
    from statsmodels.genmod.generalized_linear_model import GLMResults
    from statsmodels.discrete.discrete_model import DiscreteResults
    from statsmodels.compat.pandas import assert_frame_equal, assert_series_equal
    results = getattr(results, '_results', results)
    if isinstance(results, (GLMResults, DiscreteResults)):
        res.predict(p_exog)
        res.predict(p_exog.tolist())
        res.predict(p_exog[0].tolist())
    else:
        fitted = res.fittedvalues[:2]
        assert_allclose(fitted, res.predict(p_exog), rtol=1e-12)
        assert_allclose(fitted, res.predict(np.squeeze(p_exog).tolist()), rtol=1e-12)
        assert_allclose(fitted[:1], res.predict(p_exog[0].tolist()), rtol=1e-12)
        assert_allclose(fitted[:1], res.predict(p_exog[0]), rtol=1e-12)
        exog_index = range(len(p_exog))
        predicted = res.predict(p_exog)
        cls = pd.Series if p_exog.ndim == 1 else pd.DataFrame
        predicted_pandas = res.predict(cls(p_exog, index=exog_index))
        cls = pd.Series if predicted.ndim == 1 else pd.DataFrame
        predicted_expected = cls(predicted, index=exog_index)
        if isinstance(predicted_expected, pd.Series):
            assert_series_equal(predicted_expected, predicted_pandas)
        else:
            assert_frame_equal(predicted_expected, predicted_pandas)