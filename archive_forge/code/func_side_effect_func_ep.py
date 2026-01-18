import testtools
from unittest import mock
from troveclient.compat import auth
from troveclient.compat import exceptions
def side_effect_func_ep(attr):
    return 'test_attr_value'