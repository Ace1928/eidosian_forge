from contextlib import contextmanager
def assert_equal_sequence(x, y):
    assert type(x) is type(y)
    assert len(x) == len(y)
    assert all((x_e == y_e for x_e, y_e in zip(x, y)))