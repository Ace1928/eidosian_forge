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
def register_class(self, cls):
    """
        register a class for dumping loading
        - if it has attribute yaml_tag use that to register, else use class name
        - if it has methods to_yaml/from_yaml use those to dump/load else dump attributes
          as mapping
        """
    tag = getattr(cls, 'yaml_tag', '!' + cls.__name__)
    try:
        self.representer.add_representer(cls, cls.to_yaml)
    except AttributeError:

        def t_y(representer, data):
            return representer.represent_yaml_object(tag, data, cls, flow_style=representer.default_flow_style)
        self.representer.add_representer(cls, t_y)
    try:
        self.constructor.add_constructor(tag, cls.from_yaml)
    except AttributeError:

        def f_y(constructor, node):
            return constructor.construct_yaml_object(node, cls)
        self.constructor.add_constructor(tag, f_y)
    return cls