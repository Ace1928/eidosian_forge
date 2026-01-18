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
def move_into_pyside_package():
    import PySide2
    try:
        import PySide2.support
    except ModuleNotFoundError:
        PySide2.support = types.ModuleType('PySide2.support')
    put_into_package(PySide2.support, signature)
    put_into_package(PySide2.support.signature, mapping)
    put_into_package(PySide2.support.signature, errorhandler)
    put_into_package(PySide2.support.signature, layout)
    put_into_package(PySide2.support.signature, lib)
    put_into_package(PySide2.support.signature, parser)
    put_into_package(PySide2.support.signature.lib, enum_sig)
    put_into_package(None if orig_typing else PySide2.support.signature, typing)
    put_into_package(PySide2.support.signature, inspect)