import numpy as np
from functools import wraps
from scipy._lib._docscrape import FunctionDoc, Parameter
from scipy._lib._util import _contains_nan, AxisError, _get_nan
import inspect
def axis_nan_policy_decorator(hypotest_fun_in):

    @wraps(hypotest_fun_in)
    def axis_nan_policy_wrapper(*args, _no_deco=False, **kwds):
        if _no_deco:
            return hypotest_fun_in(*args, **kwds)
        params = list(inspect.signature(hypotest_fun_in).parameters)
        if n_samples is None:
            params = [f'arg{i}' for i in range(len(args))] + params[1:]
        maxarg = np.inf if inspect.getfullargspec(hypotest_fun_in).varargs else len(inspect.getfullargspec(hypotest_fun_in).args)
        if len(args) > maxarg:
            hypotest_fun_in(*args, **kwds)
        d_args = dict(zip(params, args))
        intersection = set(d_args) & set(kwds)
        if intersection:
            hypotest_fun_in(*args, **kwds)
        kwds.update(d_args)
        if callable(n_samples):
            n_samp = n_samples(kwds)
        else:
            n_samp = n_samples or len(args)
        n_out = n_outputs
        if callable(n_out):
            n_out = n_out(kwds)
        kwd_samp = [name for name in kwd_samples if kwds.get(name, None) is not None]
        n_kwd_samp = len(kwd_samp)
        if not kwd_samp:
            hypotest_fun_out = hypotest_fun_in
        else:

            def hypotest_fun_out(*samples, **kwds):
                new_kwds = dict(zip(kwd_samp, samples[n_samp:]))
                kwds.update(new_kwds)
                return hypotest_fun_in(*samples[:n_samp], **kwds)
        try:
            samples = [np.atleast_1d(kwds.pop(param)) for param in params[:n_samp] + kwd_samp]
        except KeyError:
            hypotest_fun_in(*args, **kwds)
        vectorized = True if 'axis' in params else False
        vectorized = vectorized and (not override['vectorization'])
        axis = kwds.pop('axis', default_axis)
        nan_policy = kwds.pop('nan_policy', 'propagate')
        keepdims = kwds.pop('keepdims', False)
        del args
        samples, sentinel = _masked_arrays_2_sentinel_arrays(samples)
        reduced_axes = axis
        if axis is None:
            if samples:
                n_dims = np.max([sample.ndim for sample in samples])
                reduced_axes = tuple(range(n_dims))
            samples = [np.asarray(sample.ravel()) for sample in samples]
        else:
            samples = _broadcast_arrays(samples, axis=axis)
            axis = np.atleast_1d(axis)
            n_axes = len(axis)
            samples = [np.moveaxis(sample, axis, range(-len(axis), 0)) for sample in samples]
            shapes = [sample.shape for sample in samples]
            new_shapes = [shape[:-n_axes] + (np.prod(shape[-n_axes:]),) for shape in shapes]
            samples = [sample.reshape(new_shape) for sample, new_shape in zip(samples, new_shapes)]
        axis = -1
        NaN = _get_nan(*samples)
        ndims = np.array([sample.ndim for sample in samples])
        if np.all(ndims <= 1):
            if nan_policy != 'propagate' or override['nan_propagation']:
                contains_nan = [_contains_nan(sample, nan_policy)[0] for sample in samples]
            else:
                contains_nan = [False] * len(samples)
            if any(contains_nan) and (nan_policy == 'propagate' and override['nan_propagation']):
                res = np.full(n_out, NaN)
                res = _add_reduced_axes(res, reduced_axes, keepdims)
                return tuple_to_result(*res)
            if any(contains_nan) and nan_policy == 'omit':
                samples = _remove_nans(samples, paired)
            if sentinel:
                samples = _remove_sentinel(samples, paired, sentinel)
            res = hypotest_fun_out(*samples, **kwds)
            res = result_to_tuple(res)
            res = _add_reduced_axes(res, reduced_axes, keepdims)
            return tuple_to_result(*res)
        empty_output = _check_empty_inputs(samples, axis)
        if empty_output is not None and (is_too_small(samples, kwds) or empty_output.size == 0):
            res = [empty_output.copy() for i in range(n_out)]
            res = _add_reduced_axes(res, reduced_axes, keepdims)
            return tuple_to_result(*res)
        lengths = np.array([sample.shape[axis] for sample in samples])
        split_indices = np.cumsum(lengths)
        x = _broadcast_concatenate(samples, axis)
        if nan_policy != 'propagate' or override['nan_propagation']:
            contains_nan, _ = _contains_nan(x, nan_policy)
        else:
            contains_nan = False
        if vectorized and (not contains_nan) and (not sentinel):
            res = hypotest_fun_out(*samples, axis=axis, **kwds)
            res = result_to_tuple(res)
            res = _add_reduced_axes(res, reduced_axes, keepdims)
            return tuple_to_result(*res)
        if contains_nan and nan_policy == 'omit':

            def hypotest_fun(x):
                samples = np.split(x, split_indices)[:n_samp + n_kwd_samp]
                samples = _remove_nans(samples, paired)
                if sentinel:
                    samples = _remove_sentinel(samples, paired, sentinel)
                if is_too_small(samples, kwds):
                    return np.full(n_out, NaN)
                return result_to_tuple(hypotest_fun_out(*samples, **kwds))
        elif contains_nan and nan_policy == 'propagate' and override['nan_propagation']:

            def hypotest_fun(x):
                if np.isnan(x).any():
                    return np.full(n_out, NaN)
                samples = np.split(x, split_indices)[:n_samp + n_kwd_samp]
                if sentinel:
                    samples = _remove_sentinel(samples, paired, sentinel)
                if is_too_small(samples, kwds):
                    return np.full(n_out, NaN)
                return result_to_tuple(hypotest_fun_out(*samples, **kwds))
        else:

            def hypotest_fun(x):
                samples = np.split(x, split_indices)[:n_samp + n_kwd_samp]
                if sentinel:
                    samples = _remove_sentinel(samples, paired, sentinel)
                if is_too_small(samples, kwds):
                    return np.full(n_out, NaN)
                return result_to_tuple(hypotest_fun_out(*samples, **kwds))
        x = np.moveaxis(x, axis, 0)
        res = np.apply_along_axis(hypotest_fun, axis=0, arr=x)
        res = _add_reduced_axes(res, reduced_axes, keepdims)
        return tuple_to_result(*res)
    _axis_parameter_doc, _axis_parameter = _get_axis_params(default_axis)
    doc = FunctionDoc(axis_nan_policy_wrapper)
    parameter_names = [param.name for param in doc['Parameters']]
    if 'axis' in parameter_names:
        doc['Parameters'][parameter_names.index('axis')] = _axis_parameter_doc
    else:
        doc['Parameters'].append(_axis_parameter_doc)
    if 'nan_policy' in parameter_names:
        doc['Parameters'][parameter_names.index('nan_policy')] = _nan_policy_parameter_doc
    else:
        doc['Parameters'].append(_nan_policy_parameter_doc)
    if 'keepdims' in parameter_names:
        doc['Parameters'][parameter_names.index('keepdims')] = _keepdims_parameter_doc
    else:
        doc['Parameters'].append(_keepdims_parameter_doc)
    doc['Notes'] += _standard_note_addition
    doc = str(doc).split('\n', 1)[1]
    axis_nan_policy_wrapper.__doc__ = str(doc)
    sig = inspect.signature(axis_nan_policy_wrapper)
    parameters = sig.parameters
    parameter_list = list(parameters.values())
    if 'axis' not in parameters:
        parameter_list.append(_axis_parameter)
    if 'nan_policy' not in parameters:
        parameter_list.append(_nan_policy_parameter)
    if 'keepdims' not in parameters:
        parameter_list.append(_keepdims_parameter)
    sig = sig.replace(parameters=parameter_list)
    axis_nan_policy_wrapper.__signature__ = sig
    return axis_nan_policy_wrapper