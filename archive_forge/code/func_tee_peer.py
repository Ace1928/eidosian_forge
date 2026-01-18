from collections import deque
from itertools import islice
from typing import (
from typing_extensions import Literal
def tee_peer(iterator: Iterator[T], buffer: Deque[T], peers: List[Deque[T]], lock: ContextManager[Any]) -> Generator[T, None, None]:
    """An individual iterator of a :py:func:`~.tee`"""
    try:
        while True:
            if not buffer:
                with lock:
                    if buffer:
                        continue
                    try:
                        item = next(iterator)
                    except StopIteration:
                        break
                    else:
                        for peer_buffer in peers:
                            peer_buffer.append(item)
            yield buffer.popleft()
    finally:
        with lock:
            for idx, peer_buffer in enumerate(peers):
                if peer_buffer is buffer:
                    peers.pop(idx)
                    break
            if not peers and hasattr(iterator, 'close'):
                iterator.close()