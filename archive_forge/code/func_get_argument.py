from ..language.ast import (FragmentDefinition, FragmentSpread,
from ..language.visitor import ParallelVisitor, TypeInfoVisitor, Visitor, visit
from ..type import GraphQLSchema
from ..utils.type_info import TypeInfo
from .rules import specified_rules
def get_argument(self):
    return self._type_info.get_argument()