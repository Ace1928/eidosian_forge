from textwrap import dedent
import pytest
from pandas import (
from pandas.io.formats.style import Styler
@pytest.mark.parametrize('func', ['apply', 'map'])
def test_apply_map_non_unique_raises(df, func):
    if func == 'apply':
        op = lambda s: ['color: red;'] * len(s)
    else:
        op = lambda v: 'color: red;'
    with pytest.raises(KeyError, match='`Styler.apply` and `.map` are not'):
        getattr(df.style, func)(op)._compute()