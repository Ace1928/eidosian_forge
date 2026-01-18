import logging
import os
import re
import sys
import warnings
from datetime import timezone
from tzlocal import utils
def get_localzone_name() -> str:
    """Get the computers configured local timezone name, if any."""
    global _cache_tz_name
    if _cache_tz_name is None:
        _cache_tz_name = _get_localzone_name()
    return _cache_tz_name