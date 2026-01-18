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
def make_dataclasses_module_callnode(pos):
    global _dataclass_loader_utilitycode
    if not _dataclass_loader_utilitycode:
        python_utility_code = UtilityCode.load_cached('Dataclasses_fallback', 'Dataclasses.py')
        python_utility_code = EncodedString(python_utility_code.impl)
        _dataclass_loader_utilitycode = TempitaUtilityCode.load('SpecificModuleLoader', 'Dataclasses.c', context={'cname': 'dataclasses', 'py_code': python_utility_code.as_c_string_literal()})
    return ExprNodes.PythonCapiCallNode(pos, '__Pyx_Load_dataclasses_Module', PyrexTypes.CFuncType(PyrexTypes.py_object_type, []), utility_code=_dataclass_loader_utilitycode, args=[])