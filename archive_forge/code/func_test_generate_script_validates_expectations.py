import email
import email.errors
import os
import re
import sysconfig
import tempfile
import textwrap
import fixtures
import pkg_resources
import six
import testscenarios
import testtools
from testtools import matchers
import virtualenv
from wheel import wheelfile
from pbr import git
from pbr import packaging
from pbr.tests import base
def test_generate_script_validates_expectations(self):
    group = 'console_scripts'
    entry_point = pkg_resources.EntryPoint(name='test-ep', module_name='pbr.packaging')
    header = '#!/usr/bin/env fake-header\n'
    template = '%(group)s %(module_name)s %(import_target)s %(invoke_target)s'
    self.assertRaises(ValueError, packaging.generate_script, group, entry_point, header, template)
    entry_point = pkg_resources.EntryPoint(name='test-ep', module_name='pbr.packaging', attrs=('attr1', 'attr2', 'attr3'))
    self.assertRaises(ValueError, packaging.generate_script, group, entry_point, header, template)