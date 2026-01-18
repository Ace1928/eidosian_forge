import gyp.easy_xml as easy_xml
import unittest
from io import StringIO
def test_EasyXml_pretty(self):
    self.assertEqual(easy_xml.XmlToString(['test3', ['GrandParent', ['Parent1', ['Child']], ['Parent2']]], pretty=True), '<?xml version="1.0" encoding="utf-8"?>\n<test3>\n  <GrandParent>\n    <Parent1>\n      <Child/>\n    </Parent1>\n    <Parent2/>\n  </GrandParent>\n</test3>\n')