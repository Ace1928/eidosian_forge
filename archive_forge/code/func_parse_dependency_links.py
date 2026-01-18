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
def parse_dependency_links(requirements_files=None):
    if requirements_files is None:
        requirements_files = get_requirements_files()
    dependency_links = []
    for line in get_reqs_from_files(requirements_files):
        if re.match('(\\s*#)|(\\s*$)', line):
            continue
        if re.match('\\s*-[ef]\\s+', line):
            dependency_links.append(re.sub('\\s*-[ef]\\s+', '', line))
        elif re.match('^\\s*(https?|git(\\+(https|ssh))?|svn|hg)\\S*:', line):
            dependency_links.append(line)
    return dependency_links