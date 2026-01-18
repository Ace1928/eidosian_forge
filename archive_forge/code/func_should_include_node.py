from ..error import GraphQLError
from ..language import ast
from ..pyutils.default_ordered_dict import DefaultOrderedDict
from ..type.definition import GraphQLInterfaceType, GraphQLUnionType
from ..type.directives import GraphQLIncludeDirective, GraphQLSkipDirective
from ..type.introspection import (SchemaMetaFieldDef, TypeMetaFieldDef,
from ..utils.type_from_ast import type_from_ast
from .values import get_argument_values, get_variable_values
def should_include_node(ctx, directives):
    """Determines if a field should be included based on the @include and
    @skip directives, where @skip has higher precidence than @include."""
    if directives:
        skip_ast = None
        for directive in directives:
            if directive.name.value == GraphQLSkipDirective.name:
                skip_ast = directive
                break
        if skip_ast:
            args = get_argument_values(GraphQLSkipDirective.args, skip_ast.arguments, ctx.variable_values)
            if args.get('if') is True:
                return False
        include_ast = None
        for directive in directives:
            if directive.name.value == GraphQLIncludeDirective.name:
                include_ast = directive
                break
        if include_ast:
            args = get_argument_values(GraphQLIncludeDirective.args, include_ast.arguments, ctx.variable_values)
            if args.get('if') is False:
                return False
    return True