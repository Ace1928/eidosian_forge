import functools
import sys
import testtools
from testtools.matchers import Is
from fixtures import MonkeyPatch, TestWithFixtures
def test_delete_missing_attribute(self):
    fixture = MonkeyPatch('fixtures.tests._fixtures.test_monkeypatch.new_attr', MonkeyPatch.delete)
    self.assertFalse('new_attr' in globals())
    fixture.setUp()
    try:
        self.assertFalse('new_attr' in globals())
    finally:
        fixture.cleanUp()
        self.assertFalse('new_attr' in globals())