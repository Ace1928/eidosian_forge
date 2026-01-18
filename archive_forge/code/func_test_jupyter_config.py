import sys
from pathlib import Path
import pytest
from playwright.sync_api import expect
from panel.tests.util import wait_until
def test_jupyter_config():
    jp_files = (Path(sys.prefix) / 'etc' / 'jupyter').rglob('panel-client-jupyter.json')
    assert len(list(jp_files)) == 2