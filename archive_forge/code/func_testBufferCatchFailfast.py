import io
import os
import sys
import subprocess
from test import support
import unittest
import unittest.test
from unittest.test.test_result import BufferedWriter
def testBufferCatchFailfast(self):
    program = self.program
    for arg, attr in (('buffer', 'buffer'), ('failfast', 'failfast'), ('catch', 'catchbreak')):
        setattr(program, attr, None)
        program.parseArgs([None])
        self.assertIs(getattr(program, attr), False)
        false = []
        setattr(program, attr, false)
        program.parseArgs([None])
        self.assertIs(getattr(program, attr), false)
        true = [42]
        setattr(program, attr, true)
        program.parseArgs([None])
        self.assertIs(getattr(program, attr), true)
        short_opt = '-%s' % arg[0]
        long_opt = '--%s' % arg
        for opt in (short_opt, long_opt):
            setattr(program, attr, None)
            program.parseArgs([None, opt])
            self.assertIs(getattr(program, attr), True)
            setattr(program, attr, False)
            with support.captured_stderr() as stderr, self.assertRaises(SystemExit) as cm:
                program.parseArgs([None, opt])
            self.assertEqual(cm.exception.args, (2,))
            setattr(program, attr, True)
            with support.captured_stderr() as stderr, self.assertRaises(SystemExit) as cm:
                program.parseArgs([None, opt])
            self.assertEqual(cm.exception.args, (2,))