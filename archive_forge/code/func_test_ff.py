from dulwich.tests import TestCase
from ..graph import WorkList, _find_lcas, can_fast_forward
from ..repo import MemoryRepo
from .utils import make_commit
def test_ff(self):
    r = MemoryRepo()
    base = make_commit()
    c1 = make_commit(parents=[base.id])
    c2 = make_commit(parents=[c1.id])
    r.object_store.add_objects([(base, None), (c1, None), (c2, None)])
    self.assertTrue(can_fast_forward(r, c1.id, c1.id))
    self.assertTrue(can_fast_forward(r, base.id, c1.id))
    self.assertTrue(can_fast_forward(r, c1.id, c2.id))
    self.assertFalse(can_fast_forward(r, c2.id, c1.id))