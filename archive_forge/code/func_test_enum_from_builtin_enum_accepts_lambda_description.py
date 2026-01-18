from textwrap import dedent
from ..argument import Argument
from ..enum import Enum, PyEnum
from ..field import Field
from ..inputfield import InputField
from ..inputobjecttype import InputObjectType
from ..mutation import Mutation
from ..scalars import String
from ..schema import ObjectType, Schema
def test_enum_from_builtin_enum_accepts_lambda_description():

    def custom_description(value):
        if not value:
            return 'StarWars Episodes'
        return 'New Hope Episode' if value == Episode.NEWHOPE else 'Other'

    def custom_deprecation_reason(value):
        return 'meh' if value == Episode.NEWHOPE else None
    PyEpisode = PyEnum('PyEpisode', 'NEWHOPE,EMPIRE,JEDI')
    Episode = Enum.from_enum(PyEpisode, description=custom_description, deprecation_reason=custom_deprecation_reason)

    class Query(ObjectType):
        foo = Episode()
    schema = Schema(query=Query).graphql_schema
    episode = schema.get_type('PyEpisode')
    assert episode.description == 'StarWars Episodes'
    assert [(name, value.description, value.deprecation_reason) for name, value in episode.values.items()] == [('NEWHOPE', 'New Hope Episode', 'meh'), ('EMPIRE', 'Other', None), ('JEDI', 'Other', None)]