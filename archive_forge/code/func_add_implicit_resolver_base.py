from __future__ import absolute_import
import re
from ruamel.yaml.compat import string_types, _DEFAULT_YAML_VERSION  # NOQA
from ruamel.yaml.error import *  # NOQA
from ruamel.yaml.nodes import MappingNode, ScalarNode, SequenceNode  # NOQA
from ruamel.yaml.util import RegExp  # NOQA
@classmethod
def add_implicit_resolver_base(cls, tag, regexp, first):
    if 'yaml_implicit_resolvers' not in cls.__dict__:
        cls.yaml_implicit_resolvers = dict(((k, cls.yaml_implicit_resolvers[k][:]) for k in cls.yaml_implicit_resolvers))
    if first is None:
        first = [None]
    for ch in first:
        cls.yaml_implicit_resolvers.setdefault(ch, []).append((tag, regexp))