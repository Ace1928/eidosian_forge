import logging
import os
import re
import sys
import warnings
from datetime import timezone
from tzlocal import utils
def get_localzone() -> zoneinfo.ZoneInfo:
    """Get the computers configured local timezone, if any."""
    global _cache_tz
    if _cache_tz is None:
        _cache_tz = _get_localzone()
    return _cache_tz