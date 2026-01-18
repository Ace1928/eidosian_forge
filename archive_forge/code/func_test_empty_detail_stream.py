from testtools import TestCase
from testtools.matchers import Contains
from fixtures import (
def test_empty_detail_stream(self):
    detail_name = 'test'
    fixture = StringStream(detail_name)
    with fixture:
        content = fixture.getDetails()[detail_name]
        self.assertEqual('', content.as_text())