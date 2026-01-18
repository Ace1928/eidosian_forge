import importlib
import os
from unittest import mock
from osc_lib.tests import utils as osc_lib_test_utils
import wrapt
from openstackclient import shell
def make_shell_wrapper(func, inst, args, kwargs):
    if 'shell_class' not in kwargs:
        kwargs['shell_class'] = shell.OpenStackShell
    return func(*args, **kwargs)