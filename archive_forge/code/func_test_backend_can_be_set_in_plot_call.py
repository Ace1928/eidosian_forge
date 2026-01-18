import sys
import types
import pytest
import pandas.util._test_decorators as td
import pandas
def test_backend_can_be_set_in_plot_call(monkeypatch, restore_backend, dummy_backend):
    monkeypatch.setitem(sys.modules, 'pandas_dummy_backend', dummy_backend)
    df = pandas.DataFrame([1, 2, 3])
    assert pandas.get_option('plotting.backend') == 'matplotlib'
    assert df.plot(backend='pandas_dummy_backend') == 'used_dummy'