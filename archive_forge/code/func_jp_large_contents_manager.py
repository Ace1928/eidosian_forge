import json
from pathlib import Path
import pytest
from jupyter_server.services.contents.filemanager import AsyncFileContentsManager
from jupyter_server.services.contents.largefilemanager import AsyncLargeFileManager
@pytest.fixture()
def jp_large_contents_manager(tmp_path):
    """Returns an AsyncLargeFileManager instance."""
    return AsyncLargeFileManager(root_dir=str(tmp_path))