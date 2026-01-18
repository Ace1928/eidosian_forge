from collections import defaultdict
from breezy import errors, foreign, ui, urlutils
def parse_git_svn_id(text):
    """Parse a git svn id string.

    :param text: git svn id
    :return: URL, revision number, uuid
    """
    head, uuid = text.rsplit(' ', 1)
    full_url, rev = head.rsplit('@', 1)
    return (full_url.encode('utf-8'), int(rev), uuid.encode('utf-8'))