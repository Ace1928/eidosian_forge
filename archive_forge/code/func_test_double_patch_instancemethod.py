import functools
import sys
import testtools
from testtools.matchers import Is
from fixtures import MonkeyPatch, TestWithFixtures
def test_double_patch_instancemethod(self):
    oldmethod = C.foo
    oldmethod_inst = C().foo
    fixture1 = MonkeyPatch('fixtures.tests._fixtures.test_monkeypatch.C.foo', fake)
    fixture2 = MonkeyPatch('fixtures.tests._fixtures.test_monkeypatch.C.foo', fake2)
    with fixture1:
        with fixture2:
            C().foo()
    self.assertEqual(oldmethod, C.foo)
    self.assertEqual(oldmethod_inst.__code__, C().foo.__code__)