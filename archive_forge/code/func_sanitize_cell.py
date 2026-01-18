from unittest import TestCase
from ipykernel.tests import utils
from nbformat.converter import convert
from nbformat.reader import reads
import re
import json
from copy import copy
import unittest
def sanitize_cell(self, cell):
    """
        remove non-reproducible things
        """
    for output in cell.outputs:
        self.strip_keys(output)
    return cell