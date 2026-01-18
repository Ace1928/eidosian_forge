import inspect
import sys
import types
import warnings
from os.path import normcase
from warnings import catch_warnings, simplefilter
from incremental import Version
from twisted.python import deprecate
from twisted.python.deprecate import (
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.python.test import deprecatedattributes
from twisted.python.test.modules_helpers import TwistedModulesMixin
from twisted.trial.unittest import SynchronousTestCase
from twisted.python.deprecate import deprecatedModuleAttribute
from incremental import Version
from twisted.python import deprecate
from twisted.python import deprecate
def test_invalidParameterType(self):
    """
        Create a fake signature with an invalid parameter
        type to test error handling.  The valid parameter
        types are specified in L{inspect.Parameter}.
        """

    class FakeSignature:

        def __init__(self, parameters):
            self.parameters = parameters

    class FakeParameter:

        def __init__(self, name, kind):
            self.name = name
            self.kind = kind

    def func(a, b):
        pass
    func(1, 2)
    parameters = inspect.signature(func).parameters
    dummyParameters = parameters.copy()
    dummyParameters['c'] = FakeParameter('fake', 'fake')
    fakeSig = FakeSignature(dummyParameters)
    self.assertRaises(TypeError, _passedSignature, fakeSig, (1, 2), {})