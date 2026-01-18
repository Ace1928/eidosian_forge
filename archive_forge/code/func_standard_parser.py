import types
import os
import string
import uuid
from paste.deploy import appconfig
from paste.script import copydir
from paste.script.command import Command, BadCommand, run as run_command
from paste.script.util import secret
from paste.util import import_string
import paste.script.templates
import pkg_resources
def standard_parser(cls, **kw):
    parser = super(AbstractInstallCommand, cls).standard_parser(**kw)
    parser.add_option('--sysconfig', action='append', dest='sysconfigs', help='System configuration file')
    parser.add_option('--no-default-sysconfig', action='store_true', dest='no_default_sysconfig', help="Don't load the default sysconfig files")
    parser.add_option('--easy-install', action='append', dest='easy_install_op', metavar='OP', help='An option to add if invoking easy_install (like --easy-install=exclude-scripts)')
    parser.add_option('--no-install', action='store_true', dest='no_install', help="Don't try to install the package (it must already be installed)")
    parser.add_option('-f', '--find-links', action='append', dest='easy_install_find_links', metavar='URL', help='Passed through to easy_install')
    return parser