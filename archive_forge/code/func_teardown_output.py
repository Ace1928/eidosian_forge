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
def teardown_output(self):
    if self._output_inited:
        self._yaml.serializer.close()
    else:
        return
    try:
        self._yaml.emitter.dispose()
    except AttributeError:
        raise
    try:
        delattr(self._yaml, '_serializer')
        delattr(self._yaml, '_emitter')
    except AttributeError:
        raise
    if self._transform:
        val = self._output.getvalue()
        if self._yaml.encoding:
            val = val.decode(self._yaml.encoding)
        if self._fstream is None:
            self._transform(val)
        else:
            self._fstream.write(self._transform(val))
            self._fstream.flush()
            self._output = self._fstream
    if self._output_path is not None:
        self._output.close()