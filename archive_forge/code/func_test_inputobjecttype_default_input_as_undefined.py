from graphql import Undefined
from ..argument import Argument
from ..field import Field
from ..inputfield import InputField
from ..inputobjecttype import InputObjectType
from ..objecttype import ObjectType
from ..scalars import Boolean, String
from ..schema import Schema
from ..unmountedtype import UnmountedType
from ... import NonNull
def test_inputobjecttype_default_input_as_undefined(set_default_input_object_type_to_undefined):

    class TestUndefinedInput(InputObjectType):
        required_field = String(required=True)
        optional_field = String()

    class Query(ObjectType):
        undefined_optionals_work = Field(NonNull(Boolean), input=TestUndefinedInput())

        def resolve_undefined_optionals_work(self, info, input: TestUndefinedInput):
            return input.required_field == 'required' and input.optional_field is Undefined
    schema = Schema(query=Query)
    result = schema.execute('query basequery {\n        undefinedOptionalsWork(input: {requiredField: "required"})\n    }\n    ')
    assert not result.errors
    assert result.data == {'undefinedOptionalsWork': True}