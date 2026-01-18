from tempfile import NamedTemporaryFile
from textwrap import dedent, indent
from unittest import TestCase, skipUnless
def test_eventual_result_wait_signature(self) -> None:
    """
        EventualResult's wait() method takes one timeout float argument.
        """
    _assert_mypy(True, dedent('                from crochet import EventualResult\n                def foo(er: EventualResult[object]) -> object:\n                    return er.wait(2.0)\n                '))
    _assert_mypy(True, dedent('                from crochet import EventualResult\n                def foo(er: EventualResult[object]) -> object:\n                    return er.wait(timeout=2.0)\n                '))