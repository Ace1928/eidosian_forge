import doctest
import unittest
def doctest_for_module(module):
    """ Create a TestCase from a module's doctests that will be run by the
        standard unittest.main().

        Example tests/test_foo.py::

            import unittest

            import foo
            from traits.testing.api import doctest_for_module

            class FooTestCase(unittest.TestCase):
                ...

            class FooDocTest(doctest_for_module(foo)):
                pass

            if __name__ == "__main__":
                # This will run and report both FooTestCase and the doctests in
                # module foo.
                unittest.main()

        Alternatively, you can say::

            FooDocTest = doctest_for_module(foo)

        instead of::

            class FooDocTest(doctest_for_module(foo)):
                pass
    """

    class C(unittest.TestCase):

        def test_dummy(self):
            pass

        def run(self, result=None):
            if hasattr(result, 'result'):
                doctest.DocTestSuite(module).run(result.result)
            else:
                doctest.DocTestSuite(module).run(result)
    return C