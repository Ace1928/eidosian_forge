from __future__ import absolute_import
import re
from ruamel.yaml.compat import string_types, _DEFAULT_YAML_VERSION  # NOQA
from ruamel.yaml.error import *  # NOQA
from ruamel.yaml.nodes import MappingNode, ScalarNode, SequenceNode  # NOQA
from ruamel.yaml.util import RegExp  # NOQA
@property
def processing_version(self):
    try:
        version = self.parser.yaml_version
    except AttributeError:
        try:
            if hasattr(self.loadumper, 'typ'):
                version = self.loadumper.version
            else:
                version = self.loadumper._serializer.use_version
        except AttributeError:
            version = None
    if version is None:
        version = self._loader_version
        if version is None:
            version = _DEFAULT_YAML_VERSION
    return version