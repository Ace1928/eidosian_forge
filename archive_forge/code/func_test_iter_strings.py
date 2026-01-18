def test_iter_strings(self, string_series):
    for i, val in enumerate(string_series):
        assert val == string_series.iloc[i]