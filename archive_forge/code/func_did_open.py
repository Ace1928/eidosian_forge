import logging
from pathlib import Path
from types import SimpleNamespace
from typing import List
import pytest
from jupyter_lsp import LanguageServerManager
from ..virtual_documents_shadow import (
def did_open(uri, text):
    return {'method': 'textDocument/didOpen', 'params': {'textDocument': {'uri': uri, 'text': text}}}