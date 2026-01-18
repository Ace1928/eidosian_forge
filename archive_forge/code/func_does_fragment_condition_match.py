from ..error import GraphQLError
from ..language import ast
from ..pyutils.default_ordered_dict import DefaultOrderedDict
from ..type.definition import GraphQLInterfaceType, GraphQLUnionType
from ..type.directives import GraphQLIncludeDirective, GraphQLSkipDirective
from ..type.introspection import (SchemaMetaFieldDef, TypeMetaFieldDef,
from ..utils.type_from_ast import type_from_ast
from .values import get_argument_values, get_variable_values
def does_fragment_condition_match(ctx, fragment, type_):
    type_condition_ast = fragment.type_condition
    if not type_condition_ast:
        return True
    conditional_type = type_from_ast(ctx.schema, type_condition_ast)
    if conditional_type.is_same_type(type_):
        return True
    if isinstance(conditional_type, (GraphQLInterfaceType, GraphQLUnionType)):
        return ctx.schema.is_possible_type(conditional_type, type_)
    return False