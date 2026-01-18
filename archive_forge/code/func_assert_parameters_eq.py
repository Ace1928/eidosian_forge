import inspect
import numpy as np
import pandas
import pytest
import modin.pandas as pd
def assert_parameters_eq(objects, attributes, allowed_different):
    pandas_obj, modin_obj = objects
    difference = []
    for m in attributes:
        if m in allowed_different:
            continue
        try:
            pandas_sig = dict(inspect.signature(getattr(pandas_obj, m)).parameters)
        except TypeError:
            continue
        try:
            modin_sig = dict(inspect.signature(getattr(modin_obj, m)).parameters)
        except TypeError:
            continue
        if not pandas_sig == modin_sig:
            append_val = (m, {i: pandas_sig[i] for i in pandas_sig.keys() if i not in modin_sig or (pandas_sig[i].default != modin_sig[i].default and (not (pandas_sig[i].default is np.nan and modin_sig[i].default is np.nan)))})
            try:
                if len(list(append_val[-1])[-1]) > 0:
                    difference.append(append_val)
            except IndexError:
                pass
    assert not len(difference), 'Missing params found in API: {}'.format(difference)
    difference = []
    for m in attributes:
        if m in allowed_different:
            continue
        try:
            pandas_sig = dict(inspect.signature(getattr(pandas_obj, m)).parameters)
        except TypeError:
            continue
        try:
            modin_sig = dict(inspect.signature(getattr(modin_obj, m)).parameters)
        except TypeError:
            continue
        if not pandas_sig == modin_sig:
            append_val = (m, {i: modin_sig[i] for i in modin_sig.keys() if i not in pandas_sig})
            try:
                if len(list(append_val[-1])[-1]) > 0:
                    difference.append(append_val)
            except IndexError:
                pass
    assert not len(difference), 'Extra params found in API: {}'.format(difference)