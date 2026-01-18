import os
import re
import time
import pytest
from wasabi.printer import Printer
from wasabi.util import MESSAGES, NO_UTF8, supports_ansi
def test_bg_color():
    p = Printer(no_print=True)
    text = 'This is a text.'
    result = p.text(text, bg_color='red')
    print(result)
    if SUPPORTS_ANSI:
        assert result == '\x1b[48;5;1mThis is a text.\x1b[0m'
    else:
        assert result == 'This is a text.'