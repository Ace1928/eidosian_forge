import os
import sys
from .. import osutils, urlutils
from ..errors import PathNotChild
from . import TestCase, TestCaseInTempDir, TestSkipped, features
def test_join_segment_parameters(self):
    join_segment_parameters = urlutils.join_segment_parameters
    self.assertEqual('/somedir/path', join_segment_parameters('/somedir/path', {}))
    self.assertEqual('/somedir/path,key1=val1', join_segment_parameters('/somedir/path', {'key1': 'val1'}))
    self.assertRaises(urlutils.InvalidURLJoin, join_segment_parameters, '/somedir/path', {'branch': 'brr,brr,brr'})
    self.assertRaises(urlutils.InvalidURLJoin, join_segment_parameters, '/somedir/path', {'key1=val1': 'val2'})
    self.assertEqual('/somedir/path,key1=val1,key2=val2', join_segment_parameters('/somedir/path', {'key1': 'val1', 'key2': 'val2'}))
    self.assertEqual('/somedir/path,key1=val1,key2=val2', join_segment_parameters('/somedir/path,key1=val1', {'key2': 'val2'}))
    self.assertEqual('/somedir/path,key1=val2', join_segment_parameters('/somedir/path,key1=val1', {'key1': 'val2'}))
    self.assertEqual('/somedir,exist=some/path,key1=val1', join_segment_parameters('/somedir,exist=some/path', {'key1': 'val1'}))
    self.assertEqual('/,key1=val1,key2=val2', join_segment_parameters('/,key1=val1', {'key2': 'val2'}))
    self.assertRaises(TypeError, join_segment_parameters, '/,key1=val1', {'foo': 42})