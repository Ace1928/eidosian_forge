from __future__ import annotations
from functools import reduce
from itertools import product
import operator
import numpy as np
import pytest
from pandas.compat import PY312
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.computation import (
from pandas.core.computation.engines import ENGINES
from pandas.core.computation.expr import (
from pandas.core.computation.expressions import (
from pandas.core.computation.ops import (
from pandas.core.computation.scope import DEFAULT_GLOBALS
@pytest.mark.parametrize('engine', ENGINES)
@pytest.mark.parametrize('parser', _parsers)
def test_disallowed_nodes(engine, parser):
    VisitorClass = _parsers[parser]
    inst = VisitorClass('x + 1', engine, parser)
    for ops in VisitorClass.unsupported_nodes:
        msg = 'nodes are not implemented'
        with pytest.raises(NotImplementedError, match=msg):
            getattr(inst, ops)()