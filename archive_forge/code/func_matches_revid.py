from ..errors import InvalidRevisionId
from ..revision import NULL_REVISION
from ..revisionspec import InvalidRevisionSpec, RevisionInfo, RevisionSpec
def matches_revid(revid):
    if revid == NULL_REVISION:
        return False
    try:
        foreign_revid, mapping = parse_revid(revid)
    except InvalidRevisionId:
        return False
    if not isinstance(mapping.vcs, ForeignGit):
        return False
    return foreign_revid.startswith(sha1)