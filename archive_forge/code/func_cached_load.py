from __future__ import annotations
import errno
import json
import re
from collections import defaultdict
from os.path import dirname
from os.path import join as pjoin
from typing import Any
def cached_load(language, domain='nbjs'):
    """Load translations for one language, using in-memory cache if available"""
    domain_cache = TRANSLATIONS_CACHE[domain]
    try:
        return domain_cache[language]
    except KeyError:
        data = load(language, domain)
        domain_cache[language] = data
        return data