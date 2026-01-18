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
def set_caption_from_template(styler, a, b):
    return styler.set_caption(f'Dataframe with a = {a} and b = {b}')