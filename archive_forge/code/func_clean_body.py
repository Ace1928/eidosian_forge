from __future__ import annotations
import logging
import os
from typing import TYPE_CHECKING
def clean_body(body: str) -> str:
    """Clean body of a message or event."""
    try:
        from bs4 import BeautifulSoup
        try:
            soup = BeautifulSoup(str(body), 'html.parser')
            body = soup.get_text()
            body = ''.join(body.splitlines())
            body = ' '.join(body.split())
            return str(body)
        except Exception:
            return str(body)
    except ImportError:
        return str(body)