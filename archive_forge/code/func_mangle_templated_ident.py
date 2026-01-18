import re
from numba.core import types
def mangle_templated_ident(identifier, parameters):
    """
    Mangle templated identifier.
    """
    template_params = 'I%sE' % ''.join(map(mangle_type_or_value, parameters)) if parameters else ''
    return mangle_identifier(identifier, template_params)