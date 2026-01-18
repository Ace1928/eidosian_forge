import os
import pathlib
import re
import shutil
import sys
import tempfile
import time
import uuid
from subprocess import PIPE, Popen
import pytest
from playwright.sync_api import expect
from panel.config import config
from panel.io.convert import BOKEH_LOCAL_WHL, PANEL_LOCAL_WHL, convert_apps
import panel as pn
import panel as pn
import panel as pn
import panel as pn
import pandas as pd
import pandas as pd
import sys
import panel as pn
import panel as pn
import panel as pn
import panel as pn
@pytest.fixture(scope='module')
def http_serve():
    temp_dir = tempfile.TemporaryDirectory()
    temp_path = pathlib.Path(temp_dir.name)
    (temp_path / 'test.html').write_text('<html><body>Test</body></html>')
    try:
        shutil.copy(PANEL_LOCAL_WHL, temp_path / PANEL_LOCAL_WHL.name)
    except shutil.SameFileError:
        pass
    try:
        shutil.copy(BOKEH_LOCAL_WHL, temp_path / BOKEH_LOCAL_WHL.name)
    except shutil.SameFileError:
        pass
    process = Popen([sys.executable, '-m', 'http.server', str(HTTP_PORT), '--directory', str(temp_path)], stdout=PIPE)
    time.sleep(10)

    def write(app):
        app_name = uuid.uuid4().hex
        app_path = temp_path / f'{app_name}.py'
        with open(app_path, 'w') as f:
            f.write(app)
        return app_path
    yield write
    process.terminate()
    process.wait()