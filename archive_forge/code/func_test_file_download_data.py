import asyncio
from io import StringIO
from pathlib import Path
import pytest
from panel.widgets import FileDownload, Progress, __file__ as wfile
def test_file_download_data():
    file_download = FileDownload(__file__, embed=True)
    tfile_data = file_download.data
    assert tfile_data is not None
    file_download.file = wfile
    assert tfile_data != file_download.data
    file_download.data = None
    file_download.embed = False
    file_download.embed = True
    assert file_download.data is not None
    file_download.data = None
    file_download._clicks += 1
    assert file_download.data is not None