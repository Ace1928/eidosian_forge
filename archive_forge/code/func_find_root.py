from collections import defaultdict
from breezy import errors, foreign, ui, urlutils
def find_root(self, uuid, url):
    for root in self._roots[uuid]:
        if url.startswith(root):
            return root
    try:
        from subvertpy.ra import RemoteAccess
    except ModuleNotFoundError:
        return None
    c = RemoteAccess(url)
    root = c.get_repos_root()
    self._roots[uuid].add(root)
    return root