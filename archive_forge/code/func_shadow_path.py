import logging
from pathlib import Path
from types import SimpleNamespace
from typing import List
import pytest
from jupyter_lsp import LanguageServerManager
from ..virtual_documents_shadow import (
@pytest.fixture
def shadow_path(tmpdir):
    return str(tmpdir.mkdir('.virtual_documents'))