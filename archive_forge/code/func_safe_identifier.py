import re
from pathlib import Path
from typing import Union
from .extern import packaging
def safe_identifier(name: str) -> str:
    """Make a string safe to be used as Python identifier.
    >>> safe_identifier("12abc")
    '_12abc'
    >>> safe_identifier("__editable__.myns.pkg-78.9.3_local")
    '__editable___myns_pkg_78_9_3_local'
    """
    safe = re.sub('\\W|^(?=\\d)', '_', name)
    assert safe.isidentifier()
    return safe