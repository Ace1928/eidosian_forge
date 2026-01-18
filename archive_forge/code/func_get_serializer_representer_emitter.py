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
def get_serializer_representer_emitter(self, stream, tlca):
    if self.Emitter is not CEmitter:
        if self.Serializer is None:
            self.Serializer = ruamel.yaml.serializer.Serializer
        self.emitter.stream = stream
        self.emitter.top_level_colon_align = tlca
        return (self.serializer, self.representer, self.emitter)
    if self.Serializer is not None:
        self.Emitter = ruamel.yaml.emitter.Emitter
        self.emitter.stream = stream
        self.emitter.top_level_colon_align = tlca
        return (self.serializer, self.representer, self.emitter)
    rslvr = ruamel.yaml.resolver.BaseResolver if self.typ == 'base' else ruamel.yaml.resolver.Resolver

    class XDumper(CEmitter, self.Representer, rslvr):

        def __init__(selfx, stream, default_style=None, default_flow_style=None, canonical=None, indent=None, width=None, allow_unicode=None, line_break=None, encoding=None, explicit_start=None, explicit_end=None, version=None, tags=None, block_seq_indent=None, top_level_colon_align=None, prefix_colon=None):
            CEmitter.__init__(selfx, stream, canonical=canonical, indent=indent, width=width, encoding=encoding, allow_unicode=allow_unicode, line_break=line_break, explicit_start=explicit_start, explicit_end=explicit_end, version=version, tags=tags)
            selfx._emitter = selfx._serializer = selfx._representer = selfx
            self.Representer.__init__(selfx, default_style=default_style, default_flow_style=default_flow_style)
            rslvr.__init__(selfx)
    self._stream = stream
    dumper = XDumper(stream, default_style=self.default_style, default_flow_style=self.default_flow_style, canonical=self.canonical, indent=self.old_indent, width=self.width, allow_unicode=self.allow_unicode, line_break=self.line_break, explicit_start=self.explicit_start, explicit_end=self.explicit_end, version=self.version, tags=self.tags)
    self._emitter = self._serializer = dumper
    return (dumper, dumper, dumper)