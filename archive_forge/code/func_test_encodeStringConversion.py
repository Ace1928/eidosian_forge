import decimal
import json
import math
import sys
import unittest
import pytest
from io import StringIO
from pathlib import Path
from srsly import ujson
def test_encodeStringConversion(self):
    input = 'A string \\ / \x08 \x0c \n \r \t </script> &'
    not_html_encoded = '"A string \\\\ \\/ \\b \\f \\n \\r \\t <\\/script> &"'
    html_encoded = '"A string \\\\ \\/ \\b \\f \\n \\r \\t \\u003c\\/script\\u003e \\u0026"'
    not_slashes_escaped = '"A string \\\\ / \\b \\f \\n \\r \\t </script> &"'

    def helper(expected_output, **encode_kwargs):
        output = ujson.encode(input, **encode_kwargs)
        self.assertEqual(output, expected_output)
        if encode_kwargs.get('escape_forward_slashes', True):
            self.assertEqual(input, json.loads(output))
            self.assertEqual(input, ujson.decode(output))
    helper(not_html_encoded, ensure_ascii=True)
    helper(not_html_encoded, ensure_ascii=False)
    helper(not_html_encoded, ensure_ascii=True, encode_html_chars=False)
    helper(not_html_encoded, ensure_ascii=False, encode_html_chars=False)
    helper(html_encoded, ensure_ascii=True, encode_html_chars=True)
    helper(html_encoded, ensure_ascii=False, encode_html_chars=True)
    helper(not_slashes_escaped, escape_forward_slashes=False)