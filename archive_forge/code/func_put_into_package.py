from __future__ import print_function, absolute_import
import sys
import os
import traceback
import types
import signature_bootstrap
from shibokensupport import signature
import shibokensupport
from shibokensupport.signature import mapping
from shibokensupport.signature import errorhandler
from shibokensupport.signature import layout
from shibokensupport.signature import lib
from shibokensupport.signature import parser
from shibokensupport.signature.lib import enum_sig
from shibokensupport.signature.parser import pyside_type_init
def put_into_package(package, module, override=None):
    name = (override if override else _get_modname(module)).rsplit('.', 1)[-1]
    if package:
        setattr(package, name, module)
    fullname = '{}.{}'.format(_get_modname(package), name) if package else name
    _set_modname(module, fullname)
    sys.modules[fullname] = module