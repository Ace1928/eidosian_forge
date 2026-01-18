import re
from numba.core import types
def mangle_identifier(ident, template_params='', *, abi_tags=(), uid=None):
    """
    Mangle the identifier with optional template parameters and abi_tags.

    Note:

    This treats '.' as '::' in C++.
    """
    if uid is not None:
        abi_tags = (f'v{uid}', *abi_tags)
    parts = [_len_encoded(_escape_string(x)) for x in ident.split('.')]
    enc_abi_tags = list(map(mangle_abi_tag, abi_tags))
    extras = template_params + ''.join(enc_abi_tags)
    if len(parts) > 1:
        return 'N%s%sE' % (''.join(parts), extras)
    else:
        return '%s%s' % (parts[0], extras)