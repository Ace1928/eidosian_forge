from pytest import raises
from ..argument import Argument
from ..dynamic import Dynamic
from ..mutation import Mutation
from ..objecttype import ObjectType
from ..scalars import String
from ..schema import Schema
from ..structures import NonNull
from ..interface import Interface
def test_mutation_allow_to_have_custom_args():

    class CreateUser(Mutation):

        class Arguments:
            name = String()
        name = String()

        def mutate(self, info, name):
            return CreateUser(name=name)

    class MyMutation(ObjectType):
        create_user = CreateUser.Field(name='createUser', description='Create a user', deprecation_reason='Is deprecated', required=True)
    field = MyMutation._meta.fields['create_user']
    assert field.name == 'createUser'
    assert field.description == 'Create a user'
    assert field.deprecation_reason == 'Is deprecated'
    assert field.type == NonNull(CreateUser)