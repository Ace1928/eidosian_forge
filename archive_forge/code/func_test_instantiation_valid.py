from statsmodels.compat.pandas import PD_LT_2_2_0, YEAR_END, is_int_index
import warnings
import numpy as np
from numpy.testing import assert_equal, assert_raises
import pandas as pd
import pytest
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.tsa.base import tsa_model
def test_instantiation_valid():
    tsa_model.__warningregistry__ = {}
    for endog in dta[:2]:
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            mod = tsa_model.TimeSeriesModel(endog)
            assert isinstance(mod._index, pd.RangeIndex) or np.issubdtype(mod._index.dtype, np.integer)
            assert_equal(mod._index_none, True)
            assert_equal(mod._index_dates, False)
            assert_equal(mod._index_generated, True)
            assert_equal(mod.data.dates, None)
            assert_equal(mod.data.freq, None)
    for endog in dta:
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            for ix, freq in date_indexes + period_indexes:
                mod = tsa_model.TimeSeriesModel(endog, dates=ix)
                if freq is None:
                    freq = ix.freq
                if not isinstance(freq, str):
                    freq = ix.freqstr
                assert_equal(isinstance(mod._index, (pd.DatetimeIndex, pd.PeriodIndex)), True)
                assert_equal(mod._index_none, False)
                assert_equal(mod._index_dates, True)
                assert_equal(mod._index_generated, False)
                assert_equal(mod._index.freq, mod._index_freq)
                assert_equal(mod.data.dates.equals(mod._index), True)
                assert_equal(mod.data.freq, freq)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            for ix, freq in date_indexes + period_indexes:
                mod = tsa_model.TimeSeriesModel(endog, dates=ix, freq=freq)
                if freq is None:
                    freq = ix.freq
                if not isinstance(freq, str):
                    freq = ix.freqstr
                assert_equal(isinstance(mod._index, (pd.DatetimeIndex, pd.PeriodIndex)), True)
                assert_equal(mod._index_none, False)
                assert_equal(mod._index_dates, True)
                assert_equal(mod._index_generated, False)
                assert_equal(mod._index.freq, mod._index_freq)
                assert_equal(mod.data.dates.equals(mod._index), True)
                assert_equal(mod.data.freq, freq)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            for ix, freq in supported_date_indexes:
                mod = tsa_model.TimeSeriesModel(endog, dates=ix, freq=freq)
                if freq is None:
                    freq = ix.freq
                if not isinstance(freq, str):
                    freq = freq.freqstr
                assert_equal(isinstance(mod._index, (pd.DatetimeIndex, pd.PeriodIndex)), True)
                assert_equal(mod._index_none, False)
                assert_equal(mod._index_dates, True)
                assert_equal(mod._index_generated, False)
                assert_equal(mod._index.freq, mod._index_freq)
                assert_equal(mod.data.dates.equals(mod._index), True)
                assert_equal(mod.data.freq, freq)
        for ix, freq in supported_increment_indexes + unsupported_indexes:
            assert_raises(ValueError, tsa_model.TimeSeriesModel, endog, dates=ix)
    for base_endog in dta[2:4]:
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            for ix, freq in date_indexes + period_indexes:
                endog = base_endog.copy()
                endog.index = ix
                mod = tsa_model.TimeSeriesModel(endog)
                if freq is None:
                    freq = ix.freq
                if not isinstance(freq, str):
                    freq = ix.freqstr
                assert_equal(isinstance(mod._index, (pd.DatetimeIndex, pd.PeriodIndex)), True)
                assert_equal(mod._index_none, False)
                assert_equal(mod._index_dates, True)
                assert_equal(mod._index_generated, False)
                assert_equal(mod._index.freq, mod._index_freq)
                assert_equal(mod.data.dates.equals(mod._index), True)
                assert_equal(mod.data.freq, freq)
        endog = base_endog.copy()
        endog.index = supported_increment_indexes[0][0]
        mod = tsa_model.TimeSeriesModel(endog)
        assert is_int_index(mod._index)
        assert_equal(mod._index_none, False)
        assert_equal(mod._index_dates, False)
        assert_equal(mod._index_generated, False)
        assert_equal(mod._index_freq, None)
        assert_equal(mod.data.dates, None)
        assert_equal(mod.data.freq, None)
        endog = base_endog.copy()
        endog.index = supported_increment_indexes[1][0]
        mod = tsa_model.TimeSeriesModel(endog)
        assert type(mod._index) is pd.RangeIndex
        assert not mod._index_none
        assert not mod._index_dates
        assert not mod._index_generated
        assert mod._index_freq is None
        assert mod.data.dates is None
        assert mod.data.freq is None
        with warnings.catch_warnings() as w:
            warnings.simplefilter('error')
            warnings.simplefilter('ignore', category=DeprecationWarning)
            for ix, freq in supported_date_indexes:
                if isinstance(ix, pd.Series) and ix.dtype == object:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        endog = pd.DataFrame(base_endog, index=ix)
                else:
                    endog = pd.DataFrame(base_endog, index=ix)
                mod = tsa_model.TimeSeriesModel(endog, freq=freq)
                if freq is None:
                    freq = ix.freq
                if not isinstance(freq, str):
                    freq = freq.freqstr
                assert_equal(isinstance(mod._index, (pd.DatetimeIndex, pd.PeriodIndex)), True)
                assert_equal(mod._index_none, False)
                assert_equal(mod._index_dates, True)
                assert_equal(mod._index_generated, False)
                assert_equal(mod._index.freq, mod._index_freq)
                assert_equal(mod.data.dates.equals(mod._index), True)
                assert_equal(mod.data.freq, freq)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            for ix, freq in supported_date_indexes:
                endog = base_endog.copy()
                endog.index = unsupported_indexes[0][0]
                mod = tsa_model.TimeSeriesModel(endog, dates=ix, freq=freq)
                if freq is None:
                    freq = ix.freq
                if not isinstance(freq, str):
                    freq = freq.freqstr
                assert_equal(isinstance(mod._index, (pd.DatetimeIndex, pd.PeriodIndex)), True)
                assert_equal(mod._index_none, False)
                assert_equal(mod._index_dates, True)
                assert_equal(mod._index_generated, False)
                assert_equal(mod._index.freq, mod._index_freq)
                assert_equal(mod.data.dates.equals(mod._index), True)
                assert_equal(mod.data.freq, freq)
        message = 'No frequency information was provided, so inferred frequency %s will be used.'
        last_len = 0
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            for ix, freq in supported_date_indexes:
                endog = base_endog.copy()
                endog.index = ix
                mod = tsa_model.TimeSeriesModel(endog)
                if freq is None:
                    freq = ix.freq
                if not isinstance(freq, str):
                    freq = freq.freqstr
                assert type(mod._index) is pd.DatetimeIndex
                assert not mod._index_none
                assert mod._index_dates
                assert not mod._index_generated
                assert_equal(mod._index.freq, mod._index_freq)
                assert mod.data.dates.equals(mod._index)
                if len(w) == last_len:
                    continue
                assert_equal(mod.data.freq.split('-')[0], freq.split('-')[0])
                assert_equal(str(w[-1].message), message % mod.data.freq)
                last_len = len(w)
        message = 'An unsupported index was provided and will be ignored when e.g. forecasting.'
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            for ix, freq in unsupported_indexes:
                endog = base_endog.copy()
                endog.index = ix
                mod = tsa_model.TimeSeriesModel(endog)
                assert_equal(isinstance(mod._index, (pd.Index, pd.RangeIndex)), True)
                assert_equal(mod._index_none, False)
                assert_equal(mod._index_dates, False)
                assert_equal(mod._index_generated, True)
                assert_equal(mod._index_freq, None)
                assert_equal(mod.data.dates, None)
                assert_equal(mod.data.freq, None)
                assert_equal(str(w[0].message), message)
        message = 'A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.'
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            for ix, freq in unsupported_date_indexes:
                endog = base_endog.copy()
                endog.index = ix
                mod = tsa_model.TimeSeriesModel(endog)
                assert isinstance(mod._index, pd.RangeIndex) or is_int_index(mod._index)
                assert_equal(mod._index_none, False)
                assert_equal(mod._index_dates, False)
                assert_equal(mod._index_generated, True)
                assert_equal(mod._index_freq, None)
                assert_equal(mod.data.dates, None)
                assert_equal(mod.data.freq, None)
                assert_equal(str(w[0].message), message)
    endog = dta[0]
    assert_raises(ValueError, tsa_model.TimeSeriesModel, endog, freq=date_indexes[1][0].freq)
    endog = dta[2].copy()
    endog.index = date_indexes[0][0]
    assert_raises(ValueError, tsa_model.TimeSeriesModel, endog, freq=date_indexes[1][0].freq)
    endog = dta[2].copy()
    endog.index = unsupported_indexes[0][0]
    assert_raises(ValueError, tsa_model.TimeSeriesModel, endog, freq=date_indexes[1][0].freq)
    endog = dta[2].copy()
    endog.index = numpy_datestr_indexes[0][0]
    assert_raises(ValueError, tsa_model.TimeSeriesModel, endog, freq=date_indexes[1][0].freq)