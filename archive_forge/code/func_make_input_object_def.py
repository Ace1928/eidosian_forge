from ..execution.values import get_argument_values
from ..language import ast
from ..pyutils.ordereddict import OrderedDict
from ..type import (GraphQLArgument, GraphQLBoolean,
from ..type.introspection import (__Directive, __DirectiveLocation,
from ..utils.value_from_ast import value_from_ast
def make_input_object_def(definition):
    return GraphQLInputObjectType(name=definition.name.value, fields=make_input_values(definition.fields, GraphQLInputObjectField))