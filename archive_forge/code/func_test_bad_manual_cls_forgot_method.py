import pytest
import cirq
def test_bad_manual_cls_forgot_method():
    with pytest.raises(TypeError, match='_value_equality_values_cls_'):

        @cirq.value_equality(manual_cls=True)
        class _:

            def _value_equality_values_(self):
                pass