import unittest
from traits.adaptation.api import reset_global_adaptation_manager
from traits.api import (
def default_foo():
    return Foo(default=True)