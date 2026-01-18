import sys
import types
from heat.engine import plugin_manager
from heat.tests import common
def args_test_mapping(*args):
    return dict(enumerate(args))