import os
import re
import sys
import inspect
import logging
from abc import ABC, ABCMeta
from fnmatch import fnmatch
from pathlib import Path
from typing import Optional, List
from jinja2 import Environment, ChoiceLoader, FileSystemLoader, \
from elementpath import datatypes
import xmlschema
from xmlschema.validators import XsdType, XsdElement, XsdAttribute
from xmlschema.names import XSD_NAMESPACE
def matching_templates(self, name):
    return self._env.list_templates(filter_func=lambda x: fnmatch(x, name))