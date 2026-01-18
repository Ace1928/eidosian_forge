import pytest
from pandas.compat._optional import VERSIONS
import pandas as pd
from pandas.core.computation import expr
from pandas.core.computation.engines import ENGINES
from pandas.util.version import Version
@pytest.mark.parametrize('engine', ENGINES)
@pytest.mark.parametrize('parser', expr.PARSERS)
def test_invalid_numexpr_version(engine, parser):
    if engine == 'numexpr':
        pytest.importorskip('numexpr')
    a, b = (1, 2)
    res = pd.eval('a + b', engine=engine, parser=parser)
    assert res == 3