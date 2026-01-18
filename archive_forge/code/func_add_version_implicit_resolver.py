from __future__ import absolute_import
import re
from ruamel.yaml.compat import string_types, _DEFAULT_YAML_VERSION  # NOQA
from ruamel.yaml.error import *  # NOQA
from ruamel.yaml.nodes import MappingNode, ScalarNode, SequenceNode  # NOQA
from ruamel.yaml.util import RegExp  # NOQA
def add_version_implicit_resolver(self, version, tag, regexp, first):
    if first is None:
        first = [None]
    impl_resolver = self._version_implicit_resolver.setdefault(version, {})
    for ch in first:
        impl_resolver.setdefault(ch, []).append((tag, regexp))