from heat.api.aws import exception as aws_exception
from heat.api.aws import utils as api_utils
from heat.common import exception as common_exception
from heat.tests import common
def test_map_remote_error_already_exists_error(self):
    ex = common_exception.StackExists(stack_name='teststack')
    expected = aws_exception.AlreadyExistsError
    self.assertIsInstance(aws_exception.map_remote_error(ex), expected)