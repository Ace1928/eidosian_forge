from __future__ import annotations
import operator
import sys
import os
import re as _re
import struct
from textwrap import fill, dedent
def rawlines(s):
    '''Return a cut-and-pastable string that, when printed, is equivalent\n    to the input. Use this when there is more than one line in the\n    string. The string returned is formatted so it can be indented\n    nicely within tests; in some cases it is wrapped in the dedent\n    function which has to be imported from textwrap.\n\n    Examples\n    ========\n\n    Note: because there are characters in the examples below that need\n    to be escaped because they are themselves within a triple quoted\n    docstring, expressions below look more complicated than they would\n    be if they were printed in an interpreter window.\n\n    >>> from sympy.utilities.misc import rawlines\n    >>> from sympy import TableForm\n    >>> s = str(TableForm([[1, 10]], headings=(None, [\'a\', \'bee\'])))\n    >>> print(rawlines(s))\n    (\n        \'a bee\\n\'\n        \'-----\\n\'\n        \'1 10 \'\n    )\n    >>> print(rawlines(\'\'\'this\n    ... that\'\'\'))\n    dedent(\'\'\'\\\n        this\n        that\'\'\')\n\n    >>> print(rawlines(\'\'\'this\n    ... that\n    ... \'\'\'))\n    dedent(\'\'\'\\\n        this\n        that\n        \'\'\')\n\n    >>> s = """this\n    ... is a triple \'\'\'\n    ... """\n    >>> print(rawlines(s))\n    dedent("""\\\n        this\n        is a triple \'\'\'\n        """)\n\n    >>> print(rawlines(\'\'\'this\n    ... that\n    ...     \'\'\'))\n    (\n        \'this\\n\'\n        \'that\\n\'\n        \'    \'\n    )\n\n    See Also\n    ========\n    filldedent, strlines\n    '''
    lines = s.split('\n')
    if len(lines) == 1:
        return repr(lines[0])
    triple = ["'''" in s, '"""' in s]
    if any((li.endswith(' ') for li in lines)) or '\\' in s or all(triple):
        rv = []
        trailing = s.endswith('\n')
        last = len(lines) - 1
        for i, li in enumerate(lines):
            if i != last or trailing:
                rv.append(repr(li + '\n'))
            else:
                rv.append(repr(li))
        return '(\n    %s\n)' % '\n    '.join(rv)
    else:
        rv = '\n    '.join(lines)
        if triple[0]:
            return 'dedent("""\\\n    %s""")' % rv
        else:
            return "dedent('''\\\n    %s''')" % rv