from __future__ import annotations
from textwrap import dedent
from pandas.core.shared_docs import _shared_docs
def create_section_header(header: str) -> str:
    """Create numpydoc section header"""
    return f'{header}\n{'-' * len(header)}\n'