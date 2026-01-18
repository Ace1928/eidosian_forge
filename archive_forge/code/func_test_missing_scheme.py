from .. import osutils, tests, urlutils
from ..directory_service import directories
from ..location import hooks as location_hooks
from ..location import location_to_url, rcp_location_to_url
def test_missing_scheme(self):
    self.skipTest('need clever guessing of scheme')
    self.assertEqual('cvs+pserver://anonymous@savi.cvs.sourceforge.net:/cvsroot/savi', location_to_url('anonymous@savi.cvs.sourceforge.net:/cvsroot/savi'))