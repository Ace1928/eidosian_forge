import os
import pytest
from wasabi.tables import row, table
from wasabi.util import supports_ansi
def test_colors_whole_table_log_friendly(data, header, footer, fg_colors, bg_colors):
    ENV_LOG_FRIENDLY = 'WASABI_LOG_FRIENDLY'
    os.environ[ENV_LOG_FRIENDLY] = 'True'
    result = table(data, header=header, footer=footer, divider=True, fg_colors=fg_colors, bg_colors=bg_colors)
    assert result == '\nCOL A            COL B   COL 3     \n--------------   -----   ----------\nHello            World   12344342  \nThis is a test   World   1234      \n--------------   -----   ----------\n                         2030203.00\n'
    del os.environ[ENV_LOG_FRIENDLY]