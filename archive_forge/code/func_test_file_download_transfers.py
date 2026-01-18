import asyncio
from io import StringIO
from pathlib import Path
import pytest
from panel.widgets import FileDownload, Progress, __file__ as wfile
def test_file_download_transfers():
    file_download = FileDownload(__file__, embed=True)
    assert file_download._transfers == 1
    file_download = FileDownload(__file__)
    assert file_download._transfers == 0
    file_download._clicks += 1
    assert file_download._transfers == 1