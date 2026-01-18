import logging
from pathlib import Path
from types import SimpleNamespace
from typing import List
import pytest
from jupyter_lsp import LanguageServerManager
from ..virtual_documents_shadow import (
def test_extract_or_none():
    obj = {'nested': {'value': 1}}
    assert extract_or_none(obj, ['nested']) == {'value': 1}
    assert extract_or_none(obj, ['nested', 'value']) == 1
    assert extract_or_none(obj, ['missing', 'value']) is None