import logging
import sys
import time
import uuid
import pytest
import panel as pn
def test_cannot_assign_string_args_with_spaces():
    terminal = pn.widgets.Terminal()
    subprocess = terminal.subprocess
    with pytest.raises(ValueError):
        subprocess.args = 'ls -l'