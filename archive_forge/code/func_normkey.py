from collections.abc import Mapping
from w3lib.http import headers_dict_to_raw
from scrapy.utils.datatypes import CaseInsensitiveDict, CaselessDict
from scrapy.utils.python import to_unicode
def normkey(self, key):
    """Normalize key to bytes"""
    return self._tobytes(key.title())