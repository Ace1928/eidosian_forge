import contextlib
@contextlib.contextmanager
def intermediate_hook(fn):
    INTERMEDIATE_HOOKS.append(fn)
    try:
        yield
    finally:
        INTERMEDIATE_HOOKS.pop()