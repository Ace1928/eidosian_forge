from ... import osutils, trace, ui
from ...errors import BzrError
from .rebase import (CommitBuilderRevisionRewriter, generate_transpose_plan,
def remove_parents(entry):
    oldrevid, (newrevid, parents) = entry
    return (oldrevid, newrevid)