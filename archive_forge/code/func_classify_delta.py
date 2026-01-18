import os.path
from ... import urlutils
from ...trace import mutter
def classify_delta(delta):
    """Determine what sort of changes a delta contains.

    :param delta: A TreeDelta to inspect
    :return: List with classes found (see classify_filename)
    """
    types = []
    for d in delta.added + delta.modified:
        types.append(classify_filename(d.path[1] or d.path[0]))
    return types