from __future__ import absolute_import, unicode_literals, print_function
import sys
import os
import warnings
import glob
from importlib import import_module
import ruamel.yaml
from ruamel.yaml.error import UnsafeLoaderWarning, YAMLError  # NOQA
from ruamel.yaml.tokens import *  # NOQA
from ruamel.yaml.events import *  # NOQA
from ruamel.yaml.nodes import *  # NOQA
from ruamel.yaml.loader import BaseLoader, SafeLoader, Loader, RoundTripLoader  # NOQA
from ruamel.yaml.dumper import BaseDumper, SafeDumper, Dumper, RoundTripDumper  # NOQA
from ruamel.yaml.compat import StringIO, BytesIO, with_metaclass, PY3, nprint
from ruamel.yaml.resolver import VersionedResolver, Resolver  # NOQA
from ruamel.yaml.representer import (
from ruamel.yaml.constructor import (
from ruamel.yaml.loader import Loader as UnsafeLoader
def init_output(self, first_data):
    if self._yaml.top_level_colon_align is True:
        tlca = max([len(str(x)) for x in first_data])
    else:
        tlca = self._yaml.top_level_colon_align
    self._yaml.get_serializer_representer_emitter(self._output, tlca)
    self._yaml.serializer.open()
    self._output_inited = True