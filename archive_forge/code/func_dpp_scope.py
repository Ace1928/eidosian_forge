import contextlib
from typing import Generator
@contextlib.contextmanager
def dpp_scope() -> Generator[None, None, None]:
    """Context manager for DPP curvature analysis

    When this scope is active, parameters are affine, not constant. The
    argument For example, if `param` is a Parameter, then

    ```
        with dpp_scope():
            print("param is constant: ", param.is_constant())
            print("param is affine: ", param.is_affine())
    ```

    would print

        param is constant: False
        param is affine: True
    """
    global _dpp_scope_active
    prev_state = _dpp_scope_active
    _dpp_scope_active = True
    yield
    _dpp_scope_active = prev_state