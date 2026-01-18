import json
from pathlib import Path
from typing import Any, List, Union
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
def remove_newlines(x: Any) -> Any:
    """Recursively remove newlines, no matter the data structure they are stored in."""
    if isinstance(x, str):
        return x.replace('\n', '')
    elif isinstance(x, list):
        return [remove_newlines(elem) for elem in x]
    elif isinstance(x, dict):
        return {k: remove_newlines(v) for k, v in x.items()}
    else:
        return x