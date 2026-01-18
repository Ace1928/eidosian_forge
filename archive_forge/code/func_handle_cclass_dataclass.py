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
def handle_cclass_dataclass(node, dataclass_args, analyse_decs_transform):
    kwargs = dict(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False, kw_only=False)
    if dataclass_args is not None:
        if dataclass_args[0]:
            error(node.pos, 'cython.dataclasses.dataclass takes no positional arguments')
        for k, v in dataclass_args[1].items():
            if k not in kwargs:
                error(node.pos, "cython.dataclasses.dataclass() got an unexpected keyword argument '%s'" % k)
            if not isinstance(v, ExprNodes.BoolNode):
                error(node.pos, 'Arguments passed to cython.dataclasses.dataclass must be True or False')
            kwargs[k] = v.value
    kw_only = kwargs['kw_only']
    fields = process_class_get_fields(node)
    dataclass_module = make_dataclasses_module_callnode(node.pos)
    dataclass_params_func = ExprNodes.AttributeNode(node.pos, obj=dataclass_module, attribute=EncodedString('_DataclassParams'))
    dataclass_params_keywords = ExprNodes.DictNode.from_pairs(node.pos, [(ExprNodes.IdentifierStringNode(node.pos, value=EncodedString(k)), ExprNodes.BoolNode(node.pos, value=v)) for k, v in kwargs.items()] + [(ExprNodes.IdentifierStringNode(node.pos, value=EncodedString(k)), ExprNodes.BoolNode(node.pos, value=v)) for k, v in [('kw_only', kw_only), ('match_args', False), ('slots', False), ('weakref_slot', False)]])
    dataclass_params = make_dataclass_call_helper(node.pos, dataclass_params_func, dataclass_params_keywords)
    dataclass_params_assignment = Nodes.SingleAssignmentNode(node.pos, lhs=ExprNodes.NameNode(node.pos, name=EncodedString('__dataclass_params__')), rhs=dataclass_params)
    dataclass_fields_stats = _set_up_dataclass_fields(node, fields, dataclass_module)
    stats = Nodes.StatListNode(node.pos, stats=[dataclass_params_assignment] + dataclass_fields_stats)
    code = TemplateCode()
    generate_init_code(code, kwargs['init'], node, fields, kw_only)
    generate_repr_code(code, kwargs['repr'], node, fields)
    generate_eq_code(code, kwargs['eq'], node, fields)
    generate_order_code(code, kwargs['order'], node, fields)
    generate_hash_code(code, kwargs['unsafe_hash'], kwargs['eq'], kwargs['frozen'], node, fields)
    stats.stats += code.generate_tree().stats
    comp_directives = Nodes.CompilerDirectivesNode(node.pos, directives=copy_inherited_directives(node.scope.directives, annotation_typing=False), body=stats)
    comp_directives.analyse_declarations(node.scope)
    analyse_decs_transform.enter_scope(node, node.scope)
    analyse_decs_transform.visit(comp_directives)
    analyse_decs_transform.exit_scope()
    node.body.stats.append(comp_directives)