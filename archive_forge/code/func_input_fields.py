from collections import OrderedDict, namedtuple
from ..language.printer import print_ast
from ..utils.ast_from_value import ast_from_value
from .definition import (GraphQLArgument, GraphQLEnumType, GraphQLEnumValue,
from .directives import DirectiveLocation
from .scalars import GraphQLBoolean, GraphQLString
@staticmethod
def input_fields(type, *_):
    if isinstance(type, GraphQLInputObjectType):
        return input_fields_to_list(type.fields)