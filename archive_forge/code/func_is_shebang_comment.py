from lib2to3.fixer_util import (FromImport, Newline, is_import,
from lib2to3.pytree import Leaf, Node
from lib2to3.pygram import python_symbols as syms
from lib2to3.pygram import token
import re
def is_shebang_comment(node):
    """
    Comments are prefixes for Leaf nodes. Returns whether the given node has a
    prefix that looks like a shebang line or an encoding line:

        #!/usr/bin/env python
        #!/usr/bin/python3
    """
    return bool(re.match(SHEBANG_REGEX, node.prefix))