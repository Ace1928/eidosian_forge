from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import ast
import difflib
import itertools
import os.path
from six import with_metaclass
import sys
import textwrap
import unittest
import pasta
from pasta.base import annotate
from pasta.base import ast_utils
from pasta.base import codegen
from pasta.base import formatting as fmt
from pasta.base import test_utils
def golden_test_generator(input_file, golden_file):

    def test(self):
        with open(input_file, 'r') as handle:
            src = handle.read()
        t = ast_utils.parse(src)
        annotator = annotate.AstAnnotator(src)
        annotator.visit(t)

        def escape(s):
            return '' if s is None else s.replace('\n', '\\n')
        result = '\n'.join(('{0:12} {1:20} \tprefix=|{2}|\tsuffix=|{3}|\tindent=|{4}|'.format(str((getattr(n, 'lineno', -1), getattr(n, 'col_offset', -1))), type(n).__name__ + ' ' + _get_node_identifier(n), escape(fmt.get(n, 'prefix')), escape(fmt.get(n, 'suffix')), escape(fmt.get(n, 'indent'))) for n in ast.walk(t))) + '\n'
        if getattr(self, 'generate_goldens', False):
            if not os.path.isdir(os.path.dirname(golden_file)):
                os.makedirs(os.path.dirname(golden_file))
            with open(golden_file, 'w') as f:
                f.write(result)
            print('Wrote: ' + golden_file)
            return
        try:
            with open(golden_file, 'r') as f:
                golden = f.read()
        except IOError:
            self.fail('Missing golden data.')
        self.assertMultiLineEqual(golden, result)
    return test