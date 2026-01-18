from collections import defaultdict
from io import BytesIO
from .. import errors, trace
from .. import transport as _mod_transport
def peel_tag(self, git_sha, default=None):
    """Peel a tag."""
    return self._re_map.get(git_sha, default)