from enum import Enum
from typing import Mapping
from .definition import (
from ..language import DirectiveLocation, print_ast
from ..pyutils import inspect
from .scalars import GraphQLBoolean, GraphQLString
@staticmethod
def specified_by_url(type_, _info):
    return getattr(type_, 'specified_by_url', None)