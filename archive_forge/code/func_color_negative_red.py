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
def color_negative_red(val):
    color = 'red' if val < 0 else 'black'
    return f'color: {color}'