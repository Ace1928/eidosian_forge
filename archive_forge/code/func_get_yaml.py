from __future__ import print_function
import sys
import pytest
def get_yaml():
    from srsly.ruamel_yaml import YAML
    return YAML()