import contextlib
import copy
import re
from textwrap import dedent
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.io.formats.style import (  # isort:skip
from pandas.io.formats.style_render import (
@pytest.mark.parametrize('deepcopy', [True, False])
def test_inherited_copy(mi_styler, deepcopy):

    class CustomStyler(Styler):
        pass
    custom_styler = CustomStyler(mi_styler.data)
    custom_styler_copy = copy.deepcopy(custom_styler) if deepcopy else copy.copy(custom_styler)
    assert isinstance(custom_styler_copy, CustomStyler)