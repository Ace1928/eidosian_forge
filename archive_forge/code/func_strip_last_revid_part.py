import os
from ....branch import Branch
from ....tests.blackbox import ExternalBase
def strip_last_revid_part(self, revid):
    """Assume revid is a revid in the default form, and strip the part
        which would be random.
        """
    return revid[:revid.rindex(b'-')]