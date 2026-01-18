import unittest
import os
import contextlib
import importlib_resources as resources
@ignore_warnings(category=DeprecationWarning)
def test_common_errors(self):
    for func in (resources.read_text, resources.read_binary, resources.open_text, resources.open_binary, resources.path, resources.is_resource, resources.contents):
        with self.subTest(func=func):
            with self.assertRaises(TypeError):
                func(None)
            with self.assertRaises((TypeError, AttributeError)):
                func(1234)
            with self.assertRaises(ModuleNotFoundError):
                func('$missing module$')