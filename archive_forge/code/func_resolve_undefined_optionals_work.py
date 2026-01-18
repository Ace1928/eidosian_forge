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
def resolve_undefined_optionals_work(self, info, input: TestUndefinedInput):
    return input.required_field == 'required' and input.optional_field is Undefined