import unittest
from ... import tests, transport, ui
from ..ui_testing import StringIOAsTTY, StringIOWithEncoding, TextUIFactory
def test_make_output_stream(self):
    output_stream = self.factory.make_output_stream()
    output_stream.write('hello!')