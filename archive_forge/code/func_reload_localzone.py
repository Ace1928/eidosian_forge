import logging
import os
import re
import sys
import warnings
from datetime import timezone
from tzlocal import utils
def reload_localzone() -> zoneinfo.ZoneInfo:
    """Reload the cached localzone. You need to call this if the timezone has changed."""
    global _cache_tz_name
    global _cache_tz
    _cache_tz_name = _get_localzone_name()
    _cache_tz = _get_localzone()
    return _cache_tz