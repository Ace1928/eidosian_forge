from textwrap import (
import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
def test_caption_as_sequence(styler):
    styler.set_caption(('full cap', 'short cap'))
    assert '<caption>full cap</caption>' in styler.to_html()