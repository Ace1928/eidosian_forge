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
def get_soabi():
    soabi = None
    try:
        soabi = sysconfig.get_config_var('SOABI')
        arch = sysconfig.get_config_var('MULTIARCH')
    except IOError:
        pass
    if soabi and arch and ('pypy' in sysconfig.get_scheme_names()):
        soabi = '%s-%s' % (soabi, arch)
    if soabi is None and 'pypy' in sysconfig.get_scheme_names():
        for suffix in get_suffixes():
            if suffix.startswith('.pypy') and suffix.endswith('.so'):
                soabi = suffix.split('.')[1]
                break
    return soabi