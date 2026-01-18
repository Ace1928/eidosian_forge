from __future__ import absolute_import
import cython
import copy
import hashlib
import sys
from . import PyrexTypes
from . import Naming
from . import ExprNodes
from . import Nodes
from . import Options
from . import Builtin
from . import Errors
from .Visitor import VisitorTransform, TreeVisitor
from .Visitor import CythonTransform, EnvTransform, ScopeTrackingTransform
from .UtilNodes import LetNode, LetRefNode
from .TreeFragment import TreeFragment
from .StringEncoding import EncodedString, _unicode
from .Errors import error, warning, CompileError, InternalError
from .Code import UtilityCode
def get_directive_class_node(self, node):
    """
        Figure out which parallel directive was used and return the associated
        Node class.

        E.g. for a cython.parallel.prange() call we return ParallelRangeNode
        """
    if self.namenode_is_cython_module:
        directive = '.'.join(self.parallel_directive)
    else:
        directive = self.parallel_directives[self.parallel_directive[0]]
        directive = '%s.%s' % (directive, '.'.join(self.parallel_directive[1:]))
        directive = directive.rstrip('.')
    cls = self.directive_to_node.get(directive)
    if cls is None and (not (self.namenode_is_cython_module and self.parallel_directive[0] != 'parallel')):
        error(node.pos, 'Invalid directive: %s' % directive)
    self.namenode_is_cython_module = False
    self.parallel_directive = None
    return cls