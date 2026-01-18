from __future__ import absolute_import
import re
from ruamel.yaml.compat import string_types, _DEFAULT_YAML_VERSION  # NOQA
from ruamel.yaml.error import *  # NOQA
from ruamel.yaml.nodes import MappingNode, ScalarNode, SequenceNode  # NOQA
from ruamel.yaml.util import RegExp  # NOQA
def get_loader_version(self, version):
    if version is None or isinstance(version, tuple):
        return version
    if isinstance(version, list):
        return tuple(version)
    return tuple(map(int, version.split(u'.')))