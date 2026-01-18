import cirq.testing
def test_assert_repr_pretty_contains():

    class TestClass:

        def _repr_pretty_(self, p, cycle):
            p.text('TestClass' if cycle else "I'm so pretty")
    cirq.testing.assert_repr_pretty_contains(TestClass(), 'pretty')
    cirq.testing.assert_repr_pretty_contains(TestClass(), 'Test', cycle=True)

    class TestClassMultipleTexts:

        def _repr_pretty_(self, p, cycle):
            if cycle:
                p.text('TestClass')
            else:
                p.text("I'm so pretty")
                p.text(' I am')
    cirq.testing.assert_repr_pretty_contains(TestClassMultipleTexts(), 'I am')
    cirq.testing.assert_repr_pretty_contains(TestClassMultipleTexts(), 'Class', cycle=True)