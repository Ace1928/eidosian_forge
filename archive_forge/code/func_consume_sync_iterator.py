from typing import Any
from typing_extensions import Iterator, AsyncIterator
def consume_sync_iterator(iterator: Iterator[Any]) -> None:
    for _ in iterator:
        ...