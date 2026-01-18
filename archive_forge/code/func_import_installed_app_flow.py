from __future__ import annotations
import logging
import os
from typing import TYPE_CHECKING, List, Optional, Tuple
def import_installed_app_flow() -> InstalledAppFlow:
    """Import InstalledAppFlow class.

    Returns:
        InstalledAppFlow: InstalledAppFlow class.
    """
    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError:
        raise ImportError('You need to install google-auth-oauthlib to use this toolkit. Try running pip install --upgrade google-auth-oauthlib')
    return InstalledAppFlow