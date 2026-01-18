import gyp.easy_xml as easy_xml
import unittest
from io import StringIO
def test_EasyXml_simple_with_attributes(self):
    self.assertEqual(easy_xml.XmlToString(['test2', {'a': 'value1', 'b': 'value2'}]), '<?xml version="1.0" encoding="utf-8"?><test2 a="value1" b="value2"/>')