import re
from numba.core import types
def mangle_abi_tag(abi_tag: str) -> str:
    return 'B' + _len_encoded(_escape_string(abi_tag))