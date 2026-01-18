import contextlib
from typing import Iterator, Mapping, Type
@contextlib.contextmanager
def map_exceptions(map: ExceptionMapping) -> Iterator[None]:
    try:
        yield
    except Exception as exc:
        for from_exc, to_exc in map.items():
            if isinstance(exc, from_exc):
                raise to_exc(exc) from exc
        raise