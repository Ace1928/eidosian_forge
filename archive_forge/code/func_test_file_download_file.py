import asyncio
from io import StringIO
from pathlib import Path
import pytest
from panel.widgets import FileDownload, Progress, __file__ as wfile
def test_file_download_file():
    with pytest.raises(ValueError):
        FileDownload(StringIO('data'))
    with pytest.raises(ValueError):
        FileDownload(embed=True)
    with pytest.raises(FileNotFoundError):
        FileDownload('nofile', embed=True)
    with pytest.raises(ValueError):
        FileDownload(666, embed=True)
    file_download = FileDownload('nofile')
    with pytest.raises(FileNotFoundError):
        file_download._clicks += 1