import sys
from tests.base import BaseTestCase
from pyasn1.type import tag
def testSuperTagCheck3(self):
    assert self.ts1.isSuperTagSetOf(tag.TagSet((), tag.Tag(tag.tagClassUniversal, tag.tagFormatSimple, 12))), 'isSuperTagSetOf() fails'