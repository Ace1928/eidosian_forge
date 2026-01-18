import importlib.metadata
from typing import Iterator
def load_entry_points(group: str) -> None:
    for entry in entry_points_for(group):
        hook = entry.load()
        if callable(hook):
            hook()