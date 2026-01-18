import functools
import sys
import testtools
from testtools.matchers import Is
from fixtures import MonkeyPatch, TestWithFixtures
def test_delete_existing_attribute(self):
    fixture = MonkeyPatch('fixtures.tests._fixtures.test_monkeypatch.reference', MonkeyPatch.delete)
    self.assertEqual(23, reference)
    fixture.setUp()
    try:
        self.assertFalse('reference' in globals())
    finally:
        fixture.cleanUp()
        self.assertEqual(23, reference)