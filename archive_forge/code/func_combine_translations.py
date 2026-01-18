from __future__ import annotations
import errno
import json
import re
from collections import defaultdict
from os.path import dirname
from os.path import join as pjoin
from typing import Any
def combine_translations(accept_language, domain='nbjs'):
    """Combine translations for multiple accepted languages.

    Returns data re-packaged in jed1.x format.
    """
    lang_codes = parse_accept_lang_header(accept_language)
    combined: dict[str, Any] = {}
    for language in lang_codes:
        if language == 'en':
            combined.clear()
        else:
            combined.update(cached_load(language, domain))
    combined[''] = {'domain': 'nbjs'}
    return {'domain': domain, 'locale_data': {domain: combined}}