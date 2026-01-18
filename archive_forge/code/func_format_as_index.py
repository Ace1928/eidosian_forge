import itertools
import json
import pkgutil
import re
from jsonschema.compat import str_types, MutableMapping, urlsplit
def format_as_index(indices):
    """
    Construct a single string containing indexing operations for the indices.

    For example, [1, 2, "foo"] -> [1][2]["foo"]

    Arguments:

        indices (sequence):

            The indices to format.

    """
    if not indices:
        return ''
    return '[%s]' % ']['.join((repr(index) for index in indices))