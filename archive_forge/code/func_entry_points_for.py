import importlib.metadata
from typing import Iterator
def entry_points_for(group: str) -> Iterator[importlib.metadata.EntryPoint]:
    try:
        eps = importlib.metadata.entry_points(group=group)
    except TypeError:
        eps = importlib.metadata.entry_points().get(group, [])
    yield from eps