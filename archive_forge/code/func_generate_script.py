from __future__ import unicode_literals
from distutils.command import install as du_install
from distutils import log
import email
import email.errors
import os
import re
import sys
import warnings
import pkg_resources
import setuptools
from setuptools.command import develop
from setuptools.command import easy_install
from setuptools.command import egg_info
from setuptools.command import install
from setuptools.command import install_scripts
from setuptools.command import sdist
from pbr import extra_files
from pbr import git
from pbr import options
import pbr.pbr_json
from pbr import testr_command
from pbr import version
import threading
from %(module_name)s import %(import_target)s
import sys
from %(module_name)s import %(import_target)s
def generate_script(group, entry_point, header, template):
    """Generate the script based on the template.

    :param str group:
        The entry-point group name, e.g., "console_scripts".
    :param str header:
        The first line of the script, e.g., "!#/usr/bin/env python".
    :param str template:
        The script template.
    :returns:
        The templated script content
    :rtype:
        str
    """
    if not entry_point.attrs or len(entry_point.attrs) > 2:
        raise ValueError("Script targets must be of the form 'func' or 'Class.class_method'.")
    script_text = template % dict(group=group, module_name=entry_point.module_name, import_target=entry_point.attrs[0], invoke_target='.'.join(entry_point.attrs))
    return header + script_text