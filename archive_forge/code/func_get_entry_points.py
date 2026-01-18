import importlib.metadata
import os
import sys
def get_entry_points(namespace):
    if sys.version_info >= (3, 10):
        return importlib.metadata.entry_points(group=namespace)
    else:
        try:
            return importlib.metadata.entry_points().get(namespace, [])
        except AttributeError:
            return importlib.metadata.entry_points().select(group=namespace)