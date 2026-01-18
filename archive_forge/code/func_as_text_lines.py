from copy import copy
from ..osutils import contains_linebreaks, contains_whitespace, sha_strings
from ..tree import Tree
def as_text_lines(self):
    """Yield text form as a sequence of lines.

        The result is returned in utf-8, because it should be signed or
        hashed in that encoding.
        """
    r = []
    a = r.append
    a(self.long_header)
    a('revision-id: %s\n' % self.revision_id.decode('utf-8'))
    a('committer: %s\n' % self.committer)
    a('timestamp: %d\n' % self.timestamp)
    a('timezone: %d\n' % self.timezone)
    a('parents:\n')
    for parent_id in sorted(self.parent_ids):
        if contains_whitespace(parent_id):
            raise ValueError(parent_id)
        a('  %s\n' % parent_id.decode('utf-8'))
    a('message:\n')
    for l in self.message.splitlines():
        a('  %s\n' % l)
    a('inventory:\n')
    for path, ie in self._get_entries():
        a(self._entry_to_line(path, ie))
    r.extend(self._revprops_to_lines())
    return [line.encode('utf-8') for line in r]