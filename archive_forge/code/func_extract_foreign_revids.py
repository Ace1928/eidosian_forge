from collections import defaultdict
from breezy import errors, foreign, ui, urlutils
def extract_foreign_revids(rev):
    """Find ids of semi-equivalent revisions in foreign VCS'es.

    :param: Bazaar revision object
    :return: Set with semi-equivalent revisions.
    """
    ret = set()
    for extractor in _foreign_revid_extractors:
        ret.update(extractor(rev))
    return ret