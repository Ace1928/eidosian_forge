from pycadf import cadftaxonomy
from pycadf.helper import api
from pycadf.tests import base
def test_convert_req_action_with_details_invalid(self):
    detail = 123
    self.assertEqual(cadftaxonomy.ACTION_READ, api.convert_req_action('GET', detail))
    self.assertEqual(cadftaxonomy.ACTION_DELETE, api.convert_req_action('DELETE', detail))