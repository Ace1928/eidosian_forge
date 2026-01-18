from __future__ import annotations
import os
import sys
import unittest
import textwrap
from typing import Any
from . import markdown, Markdown, util
def assertMarkdownRenders(self, source, expected, expected_attrs=None, **kwargs):
    """
        Test that source Markdown text renders to expected output with given keywords.

        `expected_attrs` accepts a `dict`. Each key should be the name of an attribute
        on the `Markdown` instance and the value should be the expected value after
        the source text is parsed by Markdown. After the expected output is tested,
        the expected value for each attribute is compared against the actual
        attribute of the `Markdown` instance using `TestCase.assertEqual`.
        """
    expected_attrs = expected_attrs or {}
    kws = self.default_kwargs.copy()
    kws.update(kwargs)
    md = Markdown(**kws)
    output = md.convert(source)
    self.assertMultiLineEqual(output, expected)
    for key, value in expected_attrs.items():
        self.assertEqual(getattr(md, key), value)