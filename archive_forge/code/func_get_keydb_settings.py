from __future__ import annotations
from typing import Any, Dict, Optional, TYPE_CHECKING
def get_keydb_settings(**kwargs) -> KeyDBSettings:
    """
    Get the current KeyDB settings
    """
    global _keydb_settings
    if _keydb_settings is None:
        from aiokeydb.configs.core import KeyDBSettings
        _keydb_settings = KeyDBSettings(**kwargs)
    return _keydb_settings