import os
import pytest
from wasabi.tables import row, table
from wasabi.util import supports_ansi
@pytest.fixture()
def bg_colors():
    return ['green', '23', '']