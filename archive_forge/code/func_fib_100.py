import os
import unittest
from unittest.mock import patch
from kivy.utils import (boundary, escape_markup, format_bytes_to_human,
from kivy import utils
@reify
def fib_100(self):
    """ return 100th Fibonacci number
        This uses modern view of F sub 1 = 0, F sub 2 = 1. """
    a, b = (0, 1)
    for n in range(2, 101):
        a, b = (b, a + b)
    return b