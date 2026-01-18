from pecan.middleware.static import (StaticFileMiddleware, FileWrapper,
from pecan.tests import PecanTestCase
import os
def test_file_can_be_closed(self):
    result = self._request('/static_fixtures/text.txt')
    assert result.close() is None