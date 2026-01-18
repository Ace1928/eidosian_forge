from functools import partial
from pytest import raises
from ..argument import Argument, to_arguments
from ..field import Field
from ..inputfield import InputField
from ..scalars import String
from ..structures import NonNull
def test_argument_comparasion():
    arg1 = Argument(String, name='Hey', description='Desc', default_value='default', deprecation_reason='deprecated')
    arg2 = Argument(String, name='Hey', description='Desc', default_value='default', deprecation_reason='deprecated')
    assert arg1 == arg2
    assert arg1 != String()