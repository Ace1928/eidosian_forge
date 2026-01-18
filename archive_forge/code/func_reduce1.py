from unittest import mock
from oslotest import base as test_base
from oslo_policy import _checks
from oslo_policy import _parser
from oslo_policy.tests import base
@_parser.reducer('a', 'b', 'c')
@_parser.reducer('d', 'e', 'f')
def reduce1(self):
    pass