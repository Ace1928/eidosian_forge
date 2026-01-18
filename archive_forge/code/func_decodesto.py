import re
import unittest
import jsbeautifier
import six
import copy
def decodesto(self, input, expectation=None):
    if expectation is None:
        expectation = input
    self.assertMultiLineEqual(jsbeautifier.beautify(input, self.options), expectation)
    if not expectation is None:
        self.assertMultiLineEqual(jsbeautifier.beautify(expectation, self.options), expectation)
    if self.options is None or not isinstance(self.options, (dict, tuple)):
        self.options.eol = '\r\\n'
        expectation = expectation.replace('\n', '\r\n')
        self.options.disabled = True
        self.assertMultiLineEqual(jsbeautifier.beautify(input, self.options), input or '')
        self.assertMultiLineEqual(jsbeautifier.beautify('\n\n' + expectation, self.options), '\n\n' + expectation)
        self.options.disabled = False
        self.assertMultiLineEqual(jsbeautifier.beautify(input, self.options), expectation)
        if input and input.find('\n') != -1:
            input = input.replace('\n', '\r\n')
            self.assertMultiLineEqual(jsbeautifier.beautify(input, self.options), expectation)
            self.options.eol = 'auto'
            self.assertMultiLineEqual(jsbeautifier.beautify(input, self.options), expectation)
        self.options.eol = '\n'