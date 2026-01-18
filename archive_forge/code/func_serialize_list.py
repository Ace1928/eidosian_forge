from collections.abc import Iterable
import decimal
from functools import partial
from wandb_graphql.language import ast
from wandb_graphql.language.printer import print_ast
from wandb_graphql.type import (GraphQLField, GraphQLList,
from .utils import to_camel_case
def serialize_list(serializer, values):
    assert isinstance(values, Iterable), 'Expected iterable, received "{}"'.format(repr(values))
    return [serializer(v) for v in values]