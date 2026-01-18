from pytest import mark, raises
from ...types import (
from ...types.scalars import String
from ..mutation import ClientIDMutation
@staticmethod
def mutate_and_get_payload(self, info, shared='', additional_field='', client_mutation_id=None):
    edge_type = MyEdge
    return OtherMutation(name=shared + additional_field, my_node_edge=edge_type(cursor='1', node=MyNode(name='name')))