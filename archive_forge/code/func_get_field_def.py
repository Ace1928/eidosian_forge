from ..error import GraphQLError
from ..language import ast
from ..pyutils.default_ordered_dict import DefaultOrderedDict
from ..type.definition import GraphQLInterfaceType, GraphQLUnionType
from ..type.directives import GraphQLIncludeDirective, GraphQLSkipDirective
from ..type.introspection import (SchemaMetaFieldDef, TypeMetaFieldDef,
from ..utils.type_from_ast import type_from_ast
from .values import get_argument_values, get_variable_values
def get_field_def(schema, parent_type, field_name):
    """This method looks up the field on the given type defintion.
    It has special casing for the two introspection fields, __schema
    and __typename. __typename is special because it can always be
    queried as a field, even in situations where no other fields
    are allowed, like on a Union. __schema could get automatically
    added to the query type, but that would require mutating type
    definitions, which would cause issues."""
    if field_name == '__schema' and schema.get_query_type() == parent_type:
        return SchemaMetaFieldDef
    elif field_name == '__type' and schema.get_query_type() == parent_type:
        return TypeMetaFieldDef
    elif field_name == '__typename':
        return TypeNameMetaFieldDef
    return parent_type.fields.get(field_name)