import os
import re
from io import BytesIO, StringIO
from .. import (branchbuilder, errors, gpg, log, registry, revision,
def complete_revisions(l):
    """Transform the description to suit the API.

            Tests use (revno, depth) whil the API expects (revid, revno, depth).
            Since the revid is arbitrary, we just duplicate revno
            """
    return [(r, r, d) for r, d in l]