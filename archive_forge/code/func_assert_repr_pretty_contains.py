from typing import Any
def assert_repr_pretty_contains(val: Any, substr: str, cycle: bool=False):
    """Assert that the given object has a `_repr_pretty_` output that contains the given text.

    Args:
            val: The object to test.
            substr: The string that `_repr_pretty_` is expected to contain.
            cycle: The value of `cycle` passed to `_repr_pretty_`.  `cycle` represents whether
                the call is made with a potential cycle. Typically one should handle the
                `cycle` equals `True` case by returning text that does not recursively call
                the `_repr_pretty_` to break this cycle.

    Raises:
        AssertionError: If `_repr_pretty_` does not pretty print the given text.
    """
    p = FakePrinter()
    val._repr_pretty_(p, cycle=cycle)
    assert substr in p.text_pretty, f'{substr} not in {p.text_pretty}'