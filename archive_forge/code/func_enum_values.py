from collections import OrderedDict, namedtuple
from ..language.printer import print_ast
from ..utils.ast_from_value import ast_from_value
from .definition import (GraphQLArgument, GraphQLEnumType, GraphQLEnumValue,
from .directives import DirectiveLocation
from .scalars import GraphQLBoolean, GraphQLString
@staticmethod
def enum_values(type, args, *_):
    if isinstance(type, GraphQLEnumType):
        values = type.values
        if not args.get('includeDeprecated'):
            values = [v for v in values if not v.deprecation_reason]
        return values