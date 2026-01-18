from ...error import GraphQLError
from ...language import ast
from ...type.directives import DirectiveLocation
from .base import ValidationRule
def get_directive_location_for_ast_path(ancestors):
    applied_to = ancestors[-1]
    if isinstance(applied_to, ast.OperationDefinition):
        return _operation_definition_map.get(applied_to.operation)
    elif isinstance(applied_to, ast.Field):
        return DirectiveLocation.FIELD
    elif isinstance(applied_to, ast.FragmentSpread):
        return DirectiveLocation.FRAGMENT_SPREAD
    elif isinstance(applied_to, ast.InlineFragment):
        return DirectiveLocation.INLINE_FRAGMENT
    elif isinstance(applied_to, ast.FragmentDefinition):
        return DirectiveLocation.FRAGMENT_DEFINITION
    elif isinstance(applied_to, ast.SchemaDefinition):
        return DirectiveLocation.SCHEMA
    elif isinstance(applied_to, ast.ScalarTypeDefinition):
        return DirectiveLocation.SCALAR
    elif isinstance(applied_to, ast.ObjectTypeDefinition):
        return DirectiveLocation.OBJECT
    elif isinstance(applied_to, ast.FieldDefinition):
        return DirectiveLocation.FIELD_DEFINITION
    elif isinstance(applied_to, ast.InterfaceTypeDefinition):
        return DirectiveLocation.INTERFACE
    elif isinstance(applied_to, ast.UnionTypeDefinition):
        return DirectiveLocation.UNION
    elif isinstance(applied_to, ast.EnumTypeDefinition):
        return DirectiveLocation.ENUM
    elif isinstance(applied_to, ast.EnumValueDefinition):
        return DirectiveLocation.ENUM_VALUE
    elif isinstance(applied_to, ast.InputObjectTypeDefinition):
        return DirectiveLocation.INPUT_OBJECT
    elif isinstance(applied_to, ast.InputValueDefinition):
        parent_node = ancestors[-3]
        return DirectiveLocation.INPUT_FIELD_DEFINITION if isinstance(parent_node, ast.InputObjectTypeDefinition) else DirectiveLocation.ARGUMENT_DEFINITION