import pytest
import numpy as np
import pandas as pd
import sympy
import cirq
@pytest.mark.usefixtures('closefigures')
def test_curve_fit_plot_warning():
    bad_fit = cirq.experiments.T1DecayResult(data=pd.DataFrame(columns=['delay_ns', 'false_count', 'true_count'], index=range(4), data=[[100.0, 10, 0], [400.0, 10, 0], [700.0, 10, 0], [1000.0, 10, 0]]))
    with pytest.warns(RuntimeWarning, match='Optimal parameters could not be found for curve fit'):
        bad_fit.plot(include_fit=True)