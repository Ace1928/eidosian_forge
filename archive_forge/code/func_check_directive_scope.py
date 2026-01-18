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
def check_directive_scope(self, pos, directive, scope):
    legal_scopes = Options.directive_scopes.get(directive, None)
    if legal_scopes and scope not in legal_scopes:
        self.context.nonfatal_error(PostParseError(pos, 'The %s compiler directive is not allowed in %s scope' % (directive, scope)))
        return False
    else:
        if directive not in Options.directive_types:
            error(pos, "Invalid directive: '%s'." % (directive,))
        return True