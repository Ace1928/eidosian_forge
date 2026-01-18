from typing import Collection, List, Optional, Type
from ..error import GraphQLError
from ..language import DocumentNode, ParallelVisitor, visit
from ..type import GraphQLSchema, assert_valid_schema
from ..pyutils import inspect, is_collection
from ..utilities import TypeInfo, TypeInfoVisitor
from .rules import ASTValidationRule
from .specified_rules import specified_rules, specified_sdl_rules
from .validation_context import SDLValidationContext, ValidationContext
def assert_valid_sdl_extension(document_ast: DocumentNode, schema: GraphQLSchema) -> None:
    """Assert document is a valid SDL extension.

    Utility function which asserts a SDL document is valid by throwing an error if it
    is invalid.
    """
    errors = validate_sdl(document_ast, schema)
    if errors:
        raise TypeError('\n\n'.join((error.message for error in errors)))