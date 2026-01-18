from __future__ import print_function
import numpy as np
from patsy.state import Center, Standardize, center
from patsy.util import atleast_2d_column_default
def check_stateful(cls, accepts_multicolumn, input, output, *args, **kwargs):
    input = np.asarray(input)
    output = np.asarray(output)
    test_cases = [([input], output), (input, output), ([[n] for n in input], output), ([np.array(n) for n in input], output), ([np.array(input)], output), ([np.array([n]) for n in input], output), ([np.array(input)[:, None]], atleast_2d_column_default(output)), ([np.array([[n]]) for n in input], atleast_2d_column_default(output))]
    if accepts_multicolumn:
        test_cases += [([np.column_stack((input, input[::-1]))], np.column_stack((output, output[::-1]))), ([np.array([[input[i], input[-i - 1]]]) for i in range(len(input))], np.column_stack((output, output[::-1])))]
    from patsy.util import have_pandas
    if have_pandas:
        import pandas
        pandas_type = (pandas.Series, pandas.DataFrame)
        pandas_index = np.linspace(0, 1, num=len(input))
        if output.ndim == 1:
            output_1d = pandas.Series(output, index=pandas_index)
        else:
            output_1d = pandas.DataFrame(output, index=pandas_index)
        test_cases += [([pandas.Series(input, index=pandas_index)], output_1d), ([pandas.Series([x], index=[idx]) for x, idx in zip(input, pandas_index)], output_1d)]
        if accepts_multicolumn:
            input_2d_2col = np.column_stack((input, input[::-1]))
            output_2d_2col = np.column_stack((output, output[::-1]))
            output_2col_dataframe = pandas.DataFrame(output_2d_2col, index=pandas_index)
            test_cases += [([pandas.DataFrame(input_2d_2col, index=pandas_index)], output_2col_dataframe), ([pandas.DataFrame([input_2d_2col[i, :]], index=[pandas_index[i]]) for i in range(len(input))], output_2col_dataframe)]
    for input_obj, output_obj in test_cases:
        print(input_obj)
        t = cls()
        for input_chunk in input_obj:
            t.memorize_chunk(input_chunk, *args, **kwargs)
        t.memorize_finish()
        all_outputs = []
        for input_chunk in input_obj:
            output_chunk = t.transform(input_chunk, *args, **kwargs)
            if input.ndim == output.ndim:
                assert output_chunk.ndim == np.asarray(input_chunk).ndim
            all_outputs.append(output_chunk)
        if have_pandas and isinstance(all_outputs[0], pandas_type):
            all_output1 = pandas.concat(all_outputs)
            assert np.array_equal(all_output1.index, pandas_index)
        elif all_outputs[0].ndim == 0:
            all_output1 = np.array(all_outputs)
        elif all_outputs[0].ndim == 1:
            all_output1 = np.concatenate(all_outputs)
        else:
            all_output1 = np.vstack(all_outputs)
        assert all_output1.shape[0] == len(input)
        assert np.allclose(all_output1, output_obj)
        if np.asarray(input_obj[0]).ndim == 0:
            all_input = np.array(input_obj)
        elif have_pandas and isinstance(input_obj[0], pandas_type):
            all_input = pandas.concat(input_obj)
        elif np.asarray(input_obj[0]).ndim == 1:
            all_input = np.concatenate(input_obj)
        else:
            all_input = np.vstack(input_obj)
        all_output2 = t.transform(all_input, *args, **kwargs)
        if have_pandas and isinstance(input_obj[0], pandas_type):
            assert np.array_equal(all_output2.index, pandas_index)
        if input.ndim == output.ndim:
            assert all_output2.ndim == all_input.ndim
        assert np.allclose(all_output2, output_obj)