import numpy as np
from patsy.util import (have_pandas, atleast_2d_column_default,
from patsy.state import stateful_transform
def test_crs_with_specific_constraint():
    from patsy.highlevel import incr_dbuilder, build_design_matrices, dmatrix
    x = (-1.5) ** np.arange(20)
    knots_R = np.array([-2216.8378200531006, -50.4569091796875, -0.25, 33.637939453125, 1477.8918800354004])
    centering_constraint_R = np.array([[0.06491067632316848, 1.4519875239407085, -2.1947446912471946, 1.6129783104357671, 0.06486818054755007]])
    new_x = np.array([-3000.0, -200.0, 300.0, 2000.0])
    result1 = dmatrix('cr(new_x, knots=knots_R[1:-1], lower_bound=knots_R[0], upper_bound=knots_R[-1], constraints=centering_constraint_R)')
    data_chunked = [{'x': x[:10]}, {'x': x[10:]}]
    new_data = {'x': new_x}
    builder = incr_dbuilder("cr(x, df=4, constraints='center')", lambda: iter(data_chunked))
    result2 = build_design_matrices([builder], new_data)[0]
    assert np.allclose(result1, result2, rtol=1e-12, atol=0.0)