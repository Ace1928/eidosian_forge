import sys
from io import BytesIO
from typing import List, Optional
from twisted.python import _shellcomp, reflect, usage
from twisted.python.usage import CompleteFiles, CompleteList, Completer, Completions
from twisted.trial import unittest
def test_incompleteCommandLine_case2(self):
    """
        Completion still happens even if a command-line is given
        that would normally throw UsageError.

        The existence of --unknown-option prior to the subcommand
        will break subcommand detection... but we complete anyway
        """
    outputFile = BytesIO()
    self.patch(usage.Options, '_shellCompFile', outputFile)
    opts = FighterAceOptions()
    self.assertRaises(SystemExit, opts.parseOptions, ['--fokker', '--unknown-option', 'server', '--list-server', '--_shell-completion', 'zsh:5'])
    outputFile.seek(0)
    self.assertEqual(1, len(outputFile.read(1)))
    outputFile.seek(0)
    outputFile.truncate()