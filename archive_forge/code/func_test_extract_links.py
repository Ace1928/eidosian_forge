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
@pytest.mark.parametrize('arg', ['all', 'body', 'header', 'footer'])
def test_extract_links(self, arg, flavor_read_html):
    gh_13141_data = '\n          <table>\n            <tr>\n              <th>HTTP</th>\n              <th>FTP</th>\n              <th><a href="https://en.wiktionary.org/wiki/linkless">Linkless</a></th>\n            </tr>\n            <tr>\n              <td><a href="https://en.wikipedia.org/">Wikipedia</a></td>\n              <td>SURROUNDING <a href="ftp://ftp.us.debian.org/">Debian</a> TEXT</td>\n              <td>Linkless</td>\n            </tr>\n            <tfoot>\n              <tr>\n                <td><a href="https://en.wikipedia.org/wiki/Page_footer">Footer</a></td>\n                <td>\n                  Multiple <a href="1">links:</a> <a href="2">Only first captured.</a>\n                </td>\n              </tr>\n            </tfoot>\n          </table>\n          '
    gh_13141_expected = {'head_ignore': ['HTTP', 'FTP', 'Linkless'], 'head_extract': [('HTTP', None), ('FTP', None), ('Linkless', 'https://en.wiktionary.org/wiki/linkless')], 'body_ignore': ['Wikipedia', 'SURROUNDING Debian TEXT', 'Linkless'], 'body_extract': [('Wikipedia', 'https://en.wikipedia.org/'), ('SURROUNDING Debian TEXT', 'ftp://ftp.us.debian.org/'), ('Linkless', None)], 'footer_ignore': ['Footer', 'Multiple links: Only first captured.', None], 'footer_extract': [('Footer', 'https://en.wikipedia.org/wiki/Page_footer'), ('Multiple links: Only first captured.', '1'), None]}
    data_exp = gh_13141_expected['body_ignore']
    foot_exp = gh_13141_expected['footer_ignore']
    head_exp = gh_13141_expected['head_ignore']
    if arg == 'all':
        data_exp = gh_13141_expected['body_extract']
        foot_exp = gh_13141_expected['footer_extract']
        head_exp = gh_13141_expected['head_extract']
    elif arg == 'body':
        data_exp = gh_13141_expected['body_extract']
    elif arg == 'footer':
        foot_exp = gh_13141_expected['footer_extract']
    elif arg == 'header':
        head_exp = gh_13141_expected['head_extract']
    result = flavor_read_html(StringIO(gh_13141_data), extract_links=arg)[0]
    expected = DataFrame([data_exp, foot_exp], columns=head_exp)
    expected = expected.fillna(np.nan)
    tm.assert_frame_equal(result, expected)