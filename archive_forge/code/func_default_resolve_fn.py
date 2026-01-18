from ..error import GraphQLError
from ..language import ast
from ..pyutils.default_ordered_dict import DefaultOrderedDict
from ..type.definition import GraphQLInterfaceType, GraphQLUnionType
from ..type.directives import GraphQLIncludeDirective, GraphQLSkipDirective
from ..type.introspection import (SchemaMetaFieldDef, TypeMetaFieldDef,
from ..utils.type_from_ast import type_from_ast
from .values import get_argument_values, get_variable_values
def default_resolve_fn(source, args, context, info):
    """If a resolve function is not given, then a default resolve behavior is used which takes the property of the source object
    of the same name as the field and returns it as the result, or if it's a function, returns the result of calling that function."""
    name = info.field_name
    property = getattr(source, name, None)
    if callable(property):
        return property()
    return property