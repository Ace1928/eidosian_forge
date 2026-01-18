from . import ast
from ..error import GraphQLSyntaxError
from .lexer import Lexer, TokenKind, get_token_desc, get_token_kind_desc
from .source import Source
def parse_type_system_definition(parser):
    """
      TypeSystemDefinition :
        - SchemaDefinition
        - TypeDefinition
        - TypeExtensionDefinition
        - DirectiveDefinition

      TypeDefinition :
      - ScalarTypeDefinition
      - ObjectTypeDefinition
      - InterfaceTypeDefinition
      - UnionTypeDefinition
      - EnumTypeDefinition
      - InputObjectTypeDefinition
    """
    if not peek(parser, TokenKind.NAME):
        raise unexpected(parser)
    name = parser.token.value
    if name == 'schema':
        return parse_schema_definition(parser)
    elif name == 'scalar':
        return parse_scalar_type_definition(parser)
    elif name == 'type':
        return parse_object_type_definition(parser)
    elif name == 'interface':
        return parse_interface_type_definition(parser)
    elif name == 'union':
        return parse_union_type_definition(parser)
    elif name == 'enum':
        return parse_enum_type_definition(parser)
    elif name == 'input':
        return parse_input_object_type_definition(parser)
    elif name == 'extend':
        return parse_type_extension_definition(parser)
    elif name == 'directive':
        return parse_directive_definition(parser)
    raise unexpected(parser)