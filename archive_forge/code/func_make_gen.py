import copy
from unittest import mock
from keystoneauth1 import session
from oslo_utils import uuidutils
import novaclient.api_versions
import novaclient.client
import novaclient.extension
from novaclient.tests.unit import utils
import novaclient.v2.client
def make_gen(start, end):

    def f(*args, **kwargs):
        for i in range(start, end):
            yield ('name-%s' % i, i)
    return f