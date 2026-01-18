from copy import copy
from ..osutils import contains_linebreaks, contains_whitespace, sha_strings
from ..tree import Tree
def as_short_text(self):
    """Return short digest-based testament."""
    return self.short_header.encode('ascii') + b'revision-id: %s\nsha1: %s\n' % (self.revision_id, self.as_sha1())