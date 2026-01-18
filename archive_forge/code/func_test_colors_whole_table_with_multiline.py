import os
import pytest
from wasabi.tables import row, table
from wasabi.util import supports_ansi
def test_colors_whole_table_with_multiline(data, header, footer, fg_colors, bg_colors):
    result = table(data=((['Charles', 'Quinton', 'Murphy'], 'my', 'brother'), ('1', '2', '3')), fg_colors=fg_colors, bg_colors=bg_colors, multiline=True)
    if SUPPORTS_ANSI:
        assert result == '\n\x1b[48;5;2mCharles\x1b[0m   \x1b[38;5;3;48;5;23mmy\x1b[0m   \x1b[38;5;87mbrother\x1b[0m\n\x1b[48;5;2mQuinton\x1b[0m   \x1b[38;5;3;48;5;23m  \x1b[0m   \x1b[38;5;87m       \x1b[0m\n\x1b[48;5;2mMurphy \x1b[0m   \x1b[38;5;3;48;5;23m  \x1b[0m   \x1b[38;5;87m       \x1b[0m\n\x1b[48;5;2m       \x1b[0m   \x1b[38;5;3;48;5;23m  \x1b[0m   \x1b[38;5;87m       \x1b[0m\n\x1b[48;5;2m1      \x1b[0m   \x1b[38;5;3;48;5;23m2 \x1b[0m   \x1b[38;5;87m3      \x1b[0m\n'
    else:
        assert result == '\nCharles   my   brother\nQuinton               \nMurphy                \n                      \n1         2    3      \n'