def test_items_strings(self, string_series):
    for idx, val in string_series.items():
        assert val == string_series[idx]
    assert not hasattr(string_series.items(), 'reverse')