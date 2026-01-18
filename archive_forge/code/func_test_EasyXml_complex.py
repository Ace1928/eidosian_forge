import gyp.easy_xml as easy_xml
import unittest
from io import StringIO
def test_EasyXml_complex(self):
    target = '<?xml version="1.0" encoding="utf-8"?><Project><PropertyGroup Label="Globals"><ProjectGuid>{D2250C20-3A94-4FB9-AF73-11BC5B73884B}</ProjectGuid><Keyword>Win32Proj</Keyword><RootNamespace>automated_ui_tests</RootNamespace></PropertyGroup><Import Project="$(VCTargetsPath)\\Microsoft.Cpp.props"/><PropertyGroup Condition="\'$(Configuration)|$(Platform)\'==\'Debug|Win32\'" Label="Configuration"><ConfigurationType>Application</ConfigurationType><CharacterSet>Unicode</CharacterSet></PropertyGroup></Project>'
    xml = easy_xml.XmlToString(['Project', ['PropertyGroup', {'Label': 'Globals'}, ['ProjectGuid', '{D2250C20-3A94-4FB9-AF73-11BC5B73884B}'], ['Keyword', 'Win32Proj'], ['RootNamespace', 'automated_ui_tests']], ['Import', {'Project': '$(VCTargetsPath)\\Microsoft.Cpp.props'}], ['PropertyGroup', {'Condition': "'$(Configuration)|$(Platform)'=='Debug|Win32'", 'Label': 'Configuration'}, ['ConfigurationType', 'Application'], ['CharacterSet', 'Unicode']]])
    self.assertEqual(xml, target)