from textwrap import (
import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
def html_lines(foot_prefix: str):
    assert foot_prefix.endswith('_') or foot_prefix == ''
    fp = foot_prefix
    return indent(dedent(f'        <tr>\n          <th id="T_X_level0_{fp}row0" class="{fp}row_heading level0 {fp}row0" >a</th>\n          <td id="T_X_{fp}row0_col0" class="{fp}data {fp}row0 col0" >2.610000</td>\n        </tr>\n        <tr>\n          <th id="T_X_level0_{fp}row1" class="{fp}row_heading level0 {fp}row1" >b</th>\n          <td id="T_X_{fp}row1_col0" class="{fp}data {fp}row1 col0" >2.690000</td>\n        </tr>\n        '), prefix=' ' * 4)