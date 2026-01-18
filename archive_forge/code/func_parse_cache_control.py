from __future__ import annotations
import calendar
import logging
import re
import time
from email.utils import parsedate_tz
from typing import TYPE_CHECKING, Collection, Mapping
from pip._vendor.requests.structures import CaseInsensitiveDict
from pip._vendor.cachecontrol.cache import DictCache, SeparateBodyBaseCache
from pip._vendor.cachecontrol.serialize import Serializer
def parse_cache_control(self, headers: Mapping[str, str]) -> dict[str, int | None]:
    known_directives = {'max-age': (int, True), 'max-stale': (int, False), 'min-fresh': (int, True), 'no-cache': (None, False), 'no-store': (None, False), 'no-transform': (None, False), 'only-if-cached': (None, False), 'must-revalidate': (None, False), 'public': (None, False), 'private': (None, False), 'proxy-revalidate': (None, False), 's-maxage': (int, True)}
    cc_headers = headers.get('cache-control', headers.get('Cache-Control', ''))
    retval: dict[str, int | None] = {}
    for cc_directive in cc_headers.split(','):
        if not cc_directive.strip():
            continue
        parts = cc_directive.split('=', 1)
        directive = parts[0].strip()
        try:
            typ, required = known_directives[directive]
        except KeyError:
            logger.debug('Ignoring unknown cache-control directive: %s', directive)
            continue
        if not typ or not required:
            retval[directive] = None
        if typ:
            try:
                retval[directive] = typ(parts[1].strip())
            except IndexError:
                if required:
                    logger.debug('Missing value for cache-control directive: %s', directive)
            except ValueError:
                logger.debug('Invalid value for cache-control directive %s, must be %s', directive, typ.__name__)
    return retval