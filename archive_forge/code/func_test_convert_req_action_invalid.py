from pycadf import cadftaxonomy
from pycadf.helper import api
from pycadf.tests import base
def test_convert_req_action_invalid(self):
    self.assertEqual(cadftaxonomy.UNKNOWN, api.convert_req_action(124))
    self.assertEqual(cadftaxonomy.UNKNOWN, api.convert_req_action('blah'))