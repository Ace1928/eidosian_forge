import time
from . import debug, errors, osutils, revision, trace
def iter_lefthand_ancestry(self, start_key, stop_keys=None):
    if stop_keys is None:
        stop_keys = ()
    next_key = start_key

    def get_parents(key):
        try:
            return self._parents_provider.get_parent_map([key])[key]
        except KeyError:
            raise errors.RevisionNotPresent(next_key, self)
    while True:
        if next_key in stop_keys:
            return
        parents = get_parents(next_key)
        yield next_key
        if len(parents) == 0:
            return
        else:
            next_key = parents[0]