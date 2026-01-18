import asyncio
from io import StringIO
from pathlib import Path
import pytest
from panel.widgets import FileDownload, Progress, __file__ as wfile
def test_file_path_download():
    path = Path(__file__)
    file_download = FileDownload(path)
    assert file_download.filename == 'test_misc.py'
    assert file_download.label == 'Download test_misc.py'
    file_download._clicks += 1
    assert file_download.data
    assert file_download._transfers == 1