import pytest
import cirq
def test_bad_manual_cls_incompatible_args():
    with pytest.raises(ValueError, match='incompatible'):

        @cirq.value_equality(manual_cls=True, distinct_child_types=True)
        class _:
            pass