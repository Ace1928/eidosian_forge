import fractions
import pytest
import cirq
def test_fails_to_return_not_implemented_vs_unknown():

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
    ot = cirq.testing.OrderTester()
    for k in range(len(CMP_OPS)):
        for b in [False, True]:
            item = MockValue(0, *make_impls(bad_index=k, bad_result=b))
            with pytest.raises(AssertionError, match='return NotImplemented'):
                ot.add_ascending(item)
    good_impls = make_impls(bad_index=-1, bad_result=NotImplemented)
    ot.add_ascending(MockValue(0, *good_impls))
    ot.add_ascending(MockValue(1, *good_impls))