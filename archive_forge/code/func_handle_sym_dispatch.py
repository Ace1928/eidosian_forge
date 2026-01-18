from typing import List, Type
def handle_sym_dispatch(func, args, kwargs):
    global SYM_FUNCTION_MODE
    mode = sym_function_mode()
    assert mode
    SYM_FUNCTION_MODE = mode.inner
    try:
        types: List[Type] = []
        return mode.__sym_dispatch__(func, types, args, kwargs)
    finally:
        SYM_FUNCTION_MODE = mode