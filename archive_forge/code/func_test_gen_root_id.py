from ... import tests
from .. import generate_ids
def test_gen_root_id(self):
    root_id = generate_ids.gen_root_id()
    self.assertStartsWith(root_id, b'tree_root-')