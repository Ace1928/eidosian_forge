import re
import unittest
import jsbeautifier
import six
import copy
def bt(self, input, expectation=None):
    if expectation is None:
        expectation = input
    self.decodesto(input, expectation)
    self.options.test_output_raw = True
    if self.options.end_with_newline:
        self.decodesto(input, input)
    self.options.test_output_raw = False
    current_indent_size = None
    if self.options.js and self.options.js['indent_size']:
        current_indent_size = self.options.js['indent_size']
    if not current_indent_size:
        current_indent_size = self.options.indent_size
    if current_indent_size == 4 and input:
        wrapped_input = '{\n%s\n    foo = bar;\n}' % self.wrap(input)
        wrapped_expect = '{\n%s\n    foo = bar;\n}' % self.wrap(expectation)
        self.decodesto(wrapped_input, wrapped_expect)
        self.options.test_output_raw = True
        if self.options.end_with_newline:
            self.decodesto(wrapped_input, wrapped_input)
        self.options.test_output_raw = False