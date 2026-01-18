from collections import defaultdict
from io import BytesIO
from .. import errors, trace
from .. import transport as _mod_transport
def re_unpeel_tag(self, new_git_sha, old_git_sha):
    """Re-unpeel a tag.

        Bazaar can't store unpeeled refs so in order to prevent peeling
        existing tags when pushing they are "unpeeled" here.
        """
    if old_git_sha is not None and old_git_sha in self._map[new_git_sha]:
        trace.mutter('re-unpeeling %r to %r', new_git_sha, old_git_sha)
        return old_git_sha
    return new_git_sha