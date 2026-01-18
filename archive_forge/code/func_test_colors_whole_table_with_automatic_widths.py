import os
import pytest
from wasabi.tables import row, table
from wasabi.util import supports_ansi
def test_colors_whole_table_with_automatic_widths(data, header, footer, fg_colors, bg_colors):
    result = table(data, header=header, footer=footer, divider=True, fg_colors=fg_colors, bg_colors=bg_colors)
    if SUPPORTS_ANSI:
        assert result == '\n\x1b[48;5;2mCOL A         \x1b[0m   \x1b[38;5;3;48;5;23mCOL B\x1b[0m   \x1b[38;5;87mCOL 3     \x1b[0m\n\x1b[48;5;2m--------------\x1b[0m   \x1b[38;5;3;48;5;23m-----\x1b[0m   \x1b[38;5;87m----------\x1b[0m\n\x1b[48;5;2mHello         \x1b[0m   \x1b[38;5;3;48;5;23mWorld\x1b[0m   \x1b[38;5;87m12344342  \x1b[0m\n\x1b[48;5;2mThis is a test\x1b[0m   \x1b[38;5;3;48;5;23mWorld\x1b[0m   \x1b[38;5;87m1234      \x1b[0m\n\x1b[48;5;2m--------------\x1b[0m   \x1b[38;5;3;48;5;23m-----\x1b[0m   \x1b[38;5;87m----------\x1b[0m\n\x1b[48;5;2m              \x1b[0m   \x1b[38;5;3;48;5;23m     \x1b[0m   \x1b[38;5;87m2030203.00\x1b[0m\n'
    else:
        assert result == '\nCOL A            COL B   COL 3     \n--------------   -----   ----------\nHello            World   12344342  \nThis is a test   World   1234      \n--------------   -----   ----------\n                         2030203.00\n'