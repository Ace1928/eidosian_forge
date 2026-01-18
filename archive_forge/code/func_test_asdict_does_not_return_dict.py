from __future__ import absolute_import
import unittest
import simplejson as json
from simplejson.compat import StringIO
def test_asdict_does_not_return_dict(self):
    if not mock:
        if hasattr(unittest, 'SkipTest'):
            raise unittest.SkipTest('unittest.mock required')
        else:
            print('unittest.mock not available')
            return
    fake = mock.Mock()
    self.assertTrue(hasattr(fake, '_asdict'))
    self.assertTrue(callable(fake._asdict))
    self.assertFalse(isinstance(fake._asdict(), dict))
    with self.assertRaises(TypeError):
        json.dumps({23: fake}, namedtuple_as_object=True, for_json=False)