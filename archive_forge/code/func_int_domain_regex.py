import re
from ipaddress import (
from typing import (
from . import errors
from .utils import Representation, update_not_none
from .validators import constr_length_validator, str_validator
def int_domain_regex() -> Pattern[str]:
    global _int_domain_regex_cache
    if _int_domain_regex_cache is None:
        int_chunk = '[_0-9a-\\U00040000](?:[-_0-9a-\\U00040000]{0,61}[_0-9a-\\U00040000])?'
        int_domain_ending = '(?P<tld>(\\.[^\\W\\d_]{2,63})|(\\.(?:xn--)[_0-9a-z-]{2,63}))?\\.?'
        _int_domain_regex_cache = re.compile(f'(?:{int_chunk}\\.)*?{int_chunk}{int_domain_ending}', re.IGNORECASE)
    return _int_domain_regex_cache