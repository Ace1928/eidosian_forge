import fractions
import pytest
import cirq
def make_impls(bad_index: int, bad_result: bool):

    def make_impl(i, op):

        def impl(x, y):
            if isinstance(y, MockValue):
                return op(x.val, y.val)
            if bad_index == i:
                return bad_result
            return NotImplemented
        return impl
    return [make_impl(i, op) for i, op in enumerate(CMP_OPS)]