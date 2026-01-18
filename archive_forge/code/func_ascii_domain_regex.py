import re
from ipaddress import (
from typing import (
from . import errors
from .utils import Representation, update_not_none
from .validators import constr_length_validator, str_validator
def ascii_domain_regex() -> Pattern[str]:
    global _ascii_domain_regex_cache
    if _ascii_domain_regex_cache is None:
        ascii_chunk = '[_0-9a-z](?:[-_0-9a-z]{0,61}[_0-9a-z])?'
        ascii_domain_ending = '(?P<tld>\\.[a-z]{2,63})?\\.?'
        _ascii_domain_regex_cache = re.compile(f'(?:{ascii_chunk}\\.)*?{ascii_chunk}{ascii_domain_ending}', re.IGNORECASE)
    return _ascii_domain_regex_cache