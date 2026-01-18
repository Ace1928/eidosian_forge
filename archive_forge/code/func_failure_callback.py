from unittest import mock
from testtools.matchers import GreaterThan
from stevedore import extension
from stevedore import named
from stevedore.tests import utils
def failure_callback(manager, entrypoint, error):
    errors.append((manager, entrypoint, error))