from typing import Any
from cirq import protocols
def assert_specifies_has_unitary_if_unitary(val: Any) -> None:
    """Checks that unitary values can be cheaply identifies as unitary."""
    __tracebackhide__ = True
    assert not protocols.has_unitary(val) or hasattr(val, '_has_unitary_'), f"Value is unitary but doesn't specify a _has_unitary_ method that can be used to cheaply verify this fact.\n\nval: {val!r}"