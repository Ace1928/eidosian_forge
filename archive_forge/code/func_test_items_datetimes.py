def test_items_datetimes(self, datetime_series):
    for idx, val in datetime_series.items():
        assert val == datetime_series[idx]