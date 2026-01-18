def test_iter_datetimes(self, datetime_series):
    for i, val in enumerate(datetime_series):
        assert val == datetime_series.iloc[i]