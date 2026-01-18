from __future__ import annotations
import logging
import os
from typing import TYPE_CHECKING, List, Optional, Tuple
def clean_email_body(body: str) -> str:
    """Clean email body."""
    try:
        from bs4 import BeautifulSoup
        try:
            soup = BeautifulSoup(str(body), 'html.parser')
            body = soup.get_text()
            return str(body)
        except Exception as e:
            logger.error(e)
            return str(body)
    except ImportError:
        logger.warning('BeautifulSoup not installed. Skipping cleaning.')
        return str(body)