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
def test_bad_resolver_raises(engine, parser):
    cannot_resolve = (42, 3.0)
    with pytest.raises(TypeError, match='Resolver of type .+'):
        pd.eval('1 + 2', resolvers=cannot_resolve, engine=engine, parser=parser)