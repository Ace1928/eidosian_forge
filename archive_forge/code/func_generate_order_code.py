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
def generate_order_code(code, order, node, fields):
    if not order:
        return
    for op, name in [('<', '__lt__'), ('<=', '__le__'), ('>', '__gt__'), ('>=', '__ge__')]:
        generate_cmp_code(code, op, name, node, fields)