from dulwich.tests import TestCase
from ..graph import WorkList, _find_lcas, can_fast_forward
from ..repo import MemoryRepo
from .utils import make_commit
def test_WorkList(self):
    wlst = WorkList()
    wlst.add((100, 'Test Value 1'))
    wlst.add((50, 'Test Value 2'))
    wlst.add((200, 'Test Value 3'))
    self.assertTrue(wlst.get() == (200, 'Test Value 3'))
    self.assertTrue(wlst.get() == (100, 'Test Value 1'))
    wlst.add((150, 'Test Value 4'))
    self.assertTrue(wlst.get() == (150, 'Test Value 4'))
    self.assertTrue(wlst.get() == (50, 'Test Value 2'))