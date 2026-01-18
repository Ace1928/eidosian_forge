import numpy as np
from patsy.util import have_pandas, no_pickling, assert_no_pickling
from patsy.state import stateful_transform
def test_bs_compat():
    from patsy.test_state import check_stateful
    from patsy.test_splines_bs_data import R_bs_test_x, R_bs_test_data, R_bs_num_tests
    lines = R_bs_test_data.split('\n')
    tests_ran = 0
    start_idx = lines.index('--BEGIN TEST CASE--')
    while True:
        if not lines[start_idx] == '--BEGIN TEST CASE--':
            break
        start_idx += 1
        stop_idx = lines.index('--END TEST CASE--', start_idx)
        block = lines[start_idx:stop_idx]
        test_data = {}
        for line in block:
            key, value = line.split('=', 1)
            test_data[key] = value
        kwargs = {'degree': int(test_data['degree']), 'df': eval(test_data['df']), 'knots': eval(test_data['knots'])}
        if test_data['Boundary.knots'] != 'None':
            lower, upper = eval(test_data['Boundary.knots'])
            kwargs['lower_bound'] = lower
            kwargs['upper_bound'] = upper
        kwargs['include_intercept'] = test_data['intercept'] == 'TRUE'
        output = np.asarray(eval(test_data['output']))
        if kwargs['df'] is not None:
            assert output.shape[1] == kwargs['df']
        check_stateful(BS, False, R_bs_test_x, output, **kwargs)
        tests_ran += 1
        start_idx = stop_idx + 1
    assert tests_ran == R_bs_num_tests