import io
import os
import sys
import unittest
import cherrypy
from cherrypy.test import helper
def test_call_with_literal_dict(self):
    from textwrap import dedent
    conf = dedent("\n        [my]\n        value = dict(**{'foo': 'bar'})\n        ")
    fp = StringIOFromNative(conf)
    cherrypy.config.update(fp)
    self.assertEqual(cherrypy.config['my']['value'], {'foo': 'bar'})