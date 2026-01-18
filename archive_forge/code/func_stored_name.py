import json
import os
import posixpath
import re
from hashlib import md5
from urllib.parse import unquote, urldefrag, urlsplit, urlunsplit
from django.conf import STATICFILES_STORAGE_ALIAS, settings
from django.contrib.staticfiles.utils import check_settings, matches_patterns
from django.core.exceptions import ImproperlyConfigured
from django.core.files.base import ContentFile
from django.core.files.storage import FileSystemStorage, storages
from django.utils.functional import LazyObject
def stored_name(self, name):
    parsed_name = urlsplit(unquote(name))
    clean_name = parsed_name.path.strip()
    hash_key = self.hash_key(clean_name)
    cache_name = self.hashed_files.get(hash_key)
    if cache_name is None:
        if self.manifest_strict:
            raise ValueError("Missing staticfiles manifest entry for '%s'" % clean_name)
        cache_name = self.clean_name(self.hashed_name(name))
    unparsed_name = list(parsed_name)
    unparsed_name[2] = cache_name
    if '?#' in name and (not unparsed_name[3]):
        unparsed_name[2] += '?'
    return urlunsplit(unparsed_name)