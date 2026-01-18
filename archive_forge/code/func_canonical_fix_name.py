from lib2to3.fixer_util import (FromImport, Newline, is_import,
from lib2to3.pytree import Leaf, Node
from lib2to3.pygram import python_symbols as syms
from lib2to3.pygram import token
import re
def canonical_fix_name(fix, avail_fixes):
    """
    Examples:
    >>> canonical_fix_name('fix_wrap_text_literals')
    'libfuturize.fixes.fix_wrap_text_literals'
    >>> canonical_fix_name('wrap_text_literals')
    'libfuturize.fixes.fix_wrap_text_literals'
    >>> canonical_fix_name('wrap_te')
    ValueError("unknown fixer name")
    >>> canonical_fix_name('wrap')
    ValueError("ambiguous fixer name")
    """
    if '.fix_' in fix:
        return fix
    else:
        if fix.startswith('fix_'):
            fix = fix[4:]
        found = [f for f in avail_fixes if f.endswith('fix_{0}'.format(fix))]
        if len(found) > 1:
            raise ValueError('Ambiguous fixer name. Choose a fully qualified module name instead from these:\n' + '\n'.join(('  ' + myf for myf in found)))
        elif len(found) == 0:
            raise ValueError('Unknown fixer. Use --list-fixes or -l for a list.')
        return found[0]