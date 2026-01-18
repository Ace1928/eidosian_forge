from builtins import str
import sys
import os
def check_refactor(refactorer, source, expected):
    """
        Raises an AssertionError if the given
        lib2to3.refactor.RefactoringTool does not refactor 'source' into
        'expected'.

        source, expected -- strings (typically with Python code).
        """
    new = str(refactorer.refactor_string(support.reformat(source), '<string>'))
    assert support.reformat(expected) == new, "Refactoring failed: '{}' => '{}' instead of '{}'".format(source, new.strip(), expected)