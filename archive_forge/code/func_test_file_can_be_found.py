from pecan.middleware.static import (StaticFileMiddleware, FileWrapper,
from pecan.tests import PecanTestCase
import os
def test_file_can_be_found(self):
    result = self._request('/static_fixtures/text.txt')
    assert isinstance(result, FileWrapper)
    result.close()