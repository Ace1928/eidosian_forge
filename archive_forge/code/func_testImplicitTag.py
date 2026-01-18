import sys
from tests.base import BaseTestCase
from pyasn1.type import tag
def testImplicitTag(self):
    t = self.ts1.tagImplicitly(tag.Tag(tag.tagClassApplication, tag.tagFormatSimple, 14))
    assert t == tag.TagSet(tag.Tag(tag.tagClassApplication, tag.tagFormatSimple, 12), tag.Tag(tag.tagClassApplication, tag.tagFormatSimple, 14)), 'implicit tagging went wrong'