from pytest import raises
from ..argument import Argument
from ..dynamic import Dynamic
from ..mutation import Mutation
from ..objecttype import ObjectType
from ..scalars import String
from ..schema import Schema
from ..structures import NonNull
from ..interface import Interface
def test_mutation_default_args_output():

    class CreateUser(Mutation):
        """Description."""

        class Arguments:
            name = String()
        name = String()

        def mutate(self, info, name):
            return CreateUser(name=name)

    class MyMutation(ObjectType):
        create_user = CreateUser.Field()
    field = MyMutation._meta.fields['create_user']
    assert field.name is None
    assert field.description == 'Description.'
    assert field.deprecation_reason is None
    assert field.type == CreateUser