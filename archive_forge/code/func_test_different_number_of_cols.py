from collections.abc import Iterator
from functools import partial
from io import (
import os
from pathlib import Path
import re
import threading
from urllib.error import URLError
import numpy as np
import pytest
from pandas.compat import is_platform_windows
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.io.common import file_path_to_url
def test_different_number_of_cols(self, flavor_read_html):
    expected = flavor_read_html(StringIO('<table>\n                        <thead>\n                            <tr style="text-align: right;">\n                            <th></th>\n                            <th>C_l0_g0</th>\n                            <th>C_l0_g1</th>\n                            <th>C_l0_g2</th>\n                            <th>C_l0_g3</th>\n                            <th>C_l0_g4</th>\n                            </tr>\n                        </thead>\n                        <tbody>\n                            <tr>\n                            <th>R_l0_g0</th>\n                            <td> 0.763</td>\n                            <td> 0.233</td>\n                            <td> nan</td>\n                            <td> nan</td>\n                            <td> nan</td>\n                            </tr>\n                            <tr>\n                            <th>R_l0_g1</th>\n                            <td> 0.244</td>\n                            <td> 0.285</td>\n                            <td> 0.392</td>\n                            <td> 0.137</td>\n                            <td> 0.222</td>\n                            </tr>\n                        </tbody>\n                    </table>'), index_col=0)[0]
    result = flavor_read_html(StringIO('<table>\n                    <thead>\n                        <tr style="text-align: right;">\n                        <th></th>\n                        <th>C_l0_g0</th>\n                        <th>C_l0_g1</th>\n                        <th>C_l0_g2</th>\n                        <th>C_l0_g3</th>\n                        <th>C_l0_g4</th>\n                        </tr>\n                    </thead>\n                    <tbody>\n                        <tr>\n                        <th>R_l0_g0</th>\n                        <td> 0.763</td>\n                        <td> 0.233</td>\n                        </tr>\n                        <tr>\n                        <th>R_l0_g1</th>\n                        <td> 0.244</td>\n                        <td> 0.285</td>\n                        <td> 0.392</td>\n                        <td> 0.137</td>\n                        <td> 0.222</td>\n                        </tr>\n                    </tbody>\n                 </table>'), index_col=0)[0]
    tm.assert_frame_equal(result, expected)