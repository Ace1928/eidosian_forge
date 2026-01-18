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
def test_generates_c_extensions(self):
    built_package_dir = os.path.join(self.extracted_wheel_dir, 'pbr_testpackage')
    static_object_filename = 'testext.so'
    soabi = get_soabi()
    if soabi:
        static_object_filename = 'testext.{0}.so'.format(soabi)
    static_object_path = os.path.join(built_package_dir, static_object_filename)
    self.assertTrue(os.path.exists(built_package_dir))
    self.assertTrue(os.path.exists(static_object_path))