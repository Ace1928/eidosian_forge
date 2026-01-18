from collections import OrderedDict
from textwrap import dedent
import operator
from . import ExprNodes
from . import Nodes
from . import PyrexTypes
from . import Builtin
from . import Naming
from .Errors import error, warning
from .Code import UtilityCode, TempitaUtilityCode, PyxCodeWriter
from .Visitor import VisitorTransform
from .StringEncoding import EncodedString
from .TreeFragment import TreeFragment
from .ParseTreeTransforms import NormalizeTree, SkipDeclarations
from .Options import copy_inherited_directives
def coerce_to_pyobject(self, env):
    if self.arg.type.can_coerce_to_pyobject(env):
        return self.arg.coerce_to_pyobject(env)
    else:
        return self._make_string()