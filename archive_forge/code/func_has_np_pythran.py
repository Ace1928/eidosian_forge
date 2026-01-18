from __future__ import absolute_import
from .PyrexTypes import CType, CTypedefType, CStructOrUnionType
import cython
def has_np_pythran(env):
    if env is None:
        return False
    directives = getattr(env, 'directives', None)
    return directives and directives.get('np_pythran', False)