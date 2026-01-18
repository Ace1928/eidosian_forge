from nltk.internals import find_binary, find_jar
def check_binary(binary: str, **args):
    """Skip a test via `pytest.skip` if the `binary` executable is not found.
    Keyword arguments are passed to `nltk.internals.find_binary`."""
    import pytest
    try:
        find_binary(binary, **args)
    except LookupError:
        pytest.skip(f'Skipping test because the {binary} binary was not found.')