from dulwich.tests import TestCase
from ..graph import WorkList, _find_lcas, can_fast_forward
from ..repo import MemoryRepo
from .utils import make_commit
def lookup_stamp(commit_id):
    return 100