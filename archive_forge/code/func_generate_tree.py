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
def generate_tree(self, level='c_class'):
    stat_list_node = TreeFragment(self.writer.getvalue(), level=level, pipeline=[NormalizeTree(None)]).substitute(self.placeholders)
    stat_list_node.stats += self.extra_stats
    return stat_list_node