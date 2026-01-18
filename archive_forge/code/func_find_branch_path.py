from collections import defaultdict
from breezy import errors, foreign, ui, urlutils
def find_branch_path(self, uuid, url):
    root = self.find_root(uuid, url)
    if root is None:
        return None
    assert url.startswith(root)
    return url[len(root):].strip('/')