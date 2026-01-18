from pytest import raises
from ..argument import Argument
from ..dynamic import Dynamic
from ..mutation import Mutation
from ..objecttype import ObjectType
from ..scalars import String
from ..schema import Schema
from ..structures import NonNull
from ..interface import Interface
def test_generate_mutation_no_args():

    class MyMutation(Mutation):
        """Documentation"""

        def mutate(self, info, **args):
            return args
    assert issubclass(MyMutation, ObjectType)
    assert MyMutation._meta.name == 'MyMutation'
    assert MyMutation._meta.description == 'Documentation'
    resolved = MyMutation.Field().resolver(None, None, name='Peter')
    assert resolved == {'name': 'Peter'}