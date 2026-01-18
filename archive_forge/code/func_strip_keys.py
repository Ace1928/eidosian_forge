from unittest import TestCase
from ipykernel.tests import utils
from nbformat.converter import convert
from nbformat.reader import reads
import re
import json
from copy import copy
import unittest
def strip_keys(self, d):
    """
        remove keys from STRIP_KEYS to ensure comparability
        """
    for key in self.STRIP_KEYS:
        d.pop(key, None)
    return d