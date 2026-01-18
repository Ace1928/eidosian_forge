import datetime as dt
import io
import pathlib
import time
from collections import Counter
import numpy as np
import pandas as pd
import param
import pytest
import requests
from panel.io.cache import _find_hash_func, cache
from panel.io.state import set_curdoc, state
from panel.tests.util import serve_and_wait
@pytest.mark.xdist_group('cache')
def test_cache_on_undecorated_parameterized_method():

    class Model(param.Parameterized):
        data = param.Parameter(default=1)
        executions = param.Integer(default=0)

        @cache
        def expensive_calculation(self, value):
            self.executions += 1
            return 2 * value
    model = Model()
    assert model.expensive_calculation(1) == 2
    assert model.expensive_calculation(1) == 2
    assert model.executions == 1
    assert model.expensive_calculation(2) == 4
    assert model.executions == 2