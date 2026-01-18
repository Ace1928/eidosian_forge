from pytest import raises
from ..argument import Argument
from ..dynamic import Dynamic
from ..mutation import Mutation
from ..objecttype import ObjectType
from ..scalars import String
from ..schema import Schema
from ..structures import NonNull
from ..interface import Interface
def test_generate_mutation_with_meta():

    class MyMutation(Mutation):

        class Meta:
            name = 'MyOtherMutation'
            description = 'Documentation'
            interfaces = (MyType,)

        def mutate(self, info, **args):
            return args
    assert MyMutation._meta.name == 'MyOtherMutation'
    assert MyMutation._meta.description == 'Documentation'
    assert MyMutation._meta.interfaces == (MyType,)
    resolved = MyMutation.Field().resolver(None, None, name='Peter')
    assert resolved == {'name': 'Peter'}