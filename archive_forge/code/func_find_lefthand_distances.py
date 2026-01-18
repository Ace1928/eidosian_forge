import time
from . import debug, errors, osutils, revision, trace
def find_lefthand_distances(self, keys):
    """Find the distance to null for all the keys in keys.

        :param keys: keys to lookup.
        :return: A dict key->distance for all of keys.
        """
    known_revnos = []
    ghosts = []
    for key in keys:
        try:
            known_revnos.append((key, self.find_distance_to_null(key, known_revnos)))
        except errors.GhostRevisionsHaveNoRevno:
            ghosts.append(key)
    for key in ghosts:
        known_revnos.append((key, -1))
    return dict(known_revnos)