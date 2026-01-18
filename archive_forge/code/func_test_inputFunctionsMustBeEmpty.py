from functools import reduce
from unittest import TestCase
from automat._methodical import ArgSpec, _getArgNames, _getArgSpec, _filterArgs
from .. import MethodicalMachine, NoTransition
from .. import _methodical
def test_inputFunctionsMustBeEmpty(self):
    """
        The wrapped input function must have an empty body.
        """
    _methodical._empty()
    _methodical._docstring()

    class Mechanism(object):
        m = MethodicalMachine()
        with self.assertRaises(ValueError) as cm:

            @m.input()
            def input(self):
                """an input"""
                list()
        self.assertEqual(str(cm.exception), 'function body must be empty')

    class MechanismWithDocstring(object):
        m = MethodicalMachine()

        @m.input()
        def input(self):
            """an input"""

        @m.state(initial=True)
        def start(self):
            """starting state"""
        start.upon(input, enter=start, outputs=[])
    MechanismWithDocstring().input()

    class MechanismWithPass(object):
        m = MethodicalMachine()

        @m.input()
        def input(self):
            pass

        @m.state(initial=True)
        def start(self):
            """starting state"""
        start.upon(input, enter=start, outputs=[])
    MechanismWithPass().input()

    class MechanismWithDocstringAndPass(object):
        m = MethodicalMachine()

        @m.input()
        def input(self):
            """an input"""
            pass

        @m.state(initial=True)
        def start(self):
            """starting state"""
        start.upon(input, enter=start, outputs=[])
    MechanismWithDocstringAndPass().input()

    class MechanismReturnsNone(object):
        m = MethodicalMachine()

        @m.input()
        def input(self):
            return None

        @m.state(initial=True)
        def start(self):
            """starting state"""
        start.upon(input, enter=start, outputs=[])
    MechanismReturnsNone().input()

    class MechanismWithDocstringAndReturnsNone(object):
        m = MethodicalMachine()

        @m.input()
        def input(self):
            """an input"""
            return None

        @m.state(initial=True)
        def start(self):
            """starting state"""
        start.upon(input, enter=start, outputs=[])
    MechanismWithDocstringAndReturnsNone().input()