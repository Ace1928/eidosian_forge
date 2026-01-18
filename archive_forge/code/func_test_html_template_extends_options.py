from textwrap import (
import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
def test_html_template_extends_options():
    with open('pandas/io/formats/templates/html.tpl', encoding='utf-8') as file:
        result = file.read()
    assert '{% include html_style_tpl %}' in result
    assert '{% include html_table_tpl %}' in result