from textwrap import (
import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
def test_caption_arg(styler):
    result = styler.to_html(caption='foo bar')
    assert '<caption>foo bar</caption>' in result
    result = styler.to_html()
    assert '<caption>foo bar</caption>' not in result