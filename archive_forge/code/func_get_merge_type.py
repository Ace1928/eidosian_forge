import optparse
import re
from typing import Callable, Dict
from . import errors
from . import registry as _mod_registry
from . import revisionspec
def get_merge_type(typestring):
    """Attempt to find the merge class/factory associated with a string."""
    from merge import merge_types
    try:
        return merge_types[typestring][0]
    except KeyError:
        templ = '%s%%7s: %%s' % (' ' * 12)
        lines = [templ % (f[0], f[1][1]) for f in merge_types.items()]
        type_list = '\n'.join(lines)
        msg = 'No known merge type %s. Supported types are:\n%s' % (typestring, type_list)
        raise errors.CommandError(msg)