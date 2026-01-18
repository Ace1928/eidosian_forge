from itertools import chain, repeat
from textwrap import dedent, indent
from typing import TYPE_CHECKING
from unittest import TestCase
import pytest
from IPython.core.async_helpers import _should_be_async
from IPython.testing.decorators import skip_without
def test_in_func_no_error(self):
    func_contexts = []
    func_contexts.append(('func', False, dedent('\n        def f():')))
    func_contexts.append(('method', False, dedent('\n        class MyClass:\n            def __init__(self):\n        ')))
    func_contexts.append(('async-func', True, dedent('\n        async def f():')))
    func_contexts.append(('async-method', True, dedent('\n        class MyClass:\n            async def f(self):')))
    func_contexts.append(('closure', False, dedent('\n        def f():\n            def g():\n        ')))

    def nest_case(context, case):
        lines = context.strip().splitlines()
        prefix_len = 0
        for c in lines[-1]:
            if c != ' ':
                break
            prefix_len += 1
        indented_case = indent(case, ' ' * (prefix_len + 4))
        return context + '\n' + indented_case
    vals = ('return', 'yield', 'yield from (_ for _ in range(3))')
    success_tests = zip(self._get_top_level_cases(), repeat(False))
    failure_tests = zip(self._get_ry_syntax_errors(), repeat(True))
    tests = chain(success_tests, failure_tests)
    for context_name, async_func, context in func_contexts:
        for (test_name, test_case), should_fail in tests:
            nested_case = nest_case(context, test_case)
            for val in vals:
                test_id = (context_name, test_name, val)
                cell = nested_case.format(val=val)
                with self.subTest(test_id):
                    if should_fail:
                        msg = 'SyntaxError not raised for %s' % str(test_id)
                        with self.assertRaises(SyntaxError, msg=msg):
                            iprc(cell)
                            print(cell)
                    else:
                        iprc(cell)