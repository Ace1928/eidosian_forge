from ... import tests
from .. import generate_ids
def test__next_id_suffix_increments(self):
    generate_ids._gen_file_id_suffix = b'foo-'
    generate_ids._gen_file_id_serial = 1
    try:
        self.assertEqual(b'foo-2', generate_ids._next_id_suffix())
        self.assertEqual(b'foo-3', generate_ids._next_id_suffix())
        self.assertEqual(b'foo-4', generate_ids._next_id_suffix())
        self.assertEqual(b'foo-5', generate_ids._next_id_suffix())
        self.assertEqual(b'foo-6', generate_ids._next_id_suffix())
        self.assertEqual(b'foo-7', generate_ids._next_id_suffix())
        self.assertEqual(b'foo-8', generate_ids._next_id_suffix())
        self.assertEqual(b'foo-9', generate_ids._next_id_suffix())
        self.assertEqual(b'foo-10', generate_ids._next_id_suffix())
    finally:
        generate_ids._gen_file_id_suffix = None
        generate_ids._gen_file_id_serial = 0