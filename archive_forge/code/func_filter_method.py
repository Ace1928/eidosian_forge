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
def filter_method(func):
    """Marks a method for registration as template filter."""
    func.is_filter = True
    return func