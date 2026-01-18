from .. import cache_utf8
from . import TestCase
def check_encode(self, rev_id):
    rev_id_utf8 = rev_id.encode('utf-8')
    self.assertFalse(rev_id in cache_utf8._unicode_to_utf8_map)
    self.assertFalse(rev_id_utf8 in cache_utf8._utf8_to_unicode_map)
    self.assertEqual(rev_id_utf8, cache_utf8.encode(rev_id))
    self.assertTrue(rev_id in cache_utf8._unicode_to_utf8_map)
    self.assertTrue(rev_id_utf8 in cache_utf8._utf8_to_unicode_map)
    self.assertEqual(rev_id, cache_utf8.decode(rev_id_utf8))
    cache_utf8.clear_encoding_cache()
    self.assertFalse(rev_id in cache_utf8._unicode_to_utf8_map)
    self.assertFalse(rev_id_utf8 in cache_utf8._utf8_to_unicode_map)