from os import chdir, devnull, getcwd
from subprocess import PIPE, Popen
from sys import executable
from twisted.python.filepath import FilePath
from twisted.python.modules import getModule
from twisted.python.test.test_shellcomp import ZshScriptTestMixin
from twisted.trial.unittest import SkipTest, TestCase
def outputFromPythonScript(script, *args):
    """
    Synchronously run a Python script, with the same Python interpreter that
    ran the process calling this function, using L{Popen}, using the given
    command-line arguments, with standard input and standard error both
    redirected to L{os.devnull}, and return its output as a string.

    @param script: The path to the script.
    @type script: L{FilePath}

    @param args: The command-line arguments to follow the script in its
        invocation (the desired C{sys.argv[1:]}).
    @type args: L{tuple} of L{str}

    @return: the output passed to the proces's C{stdout}, without any messages
        from C{stderr}.
    @rtype: L{bytes}
    """
    with open(devnull, 'rb') as nullInput, open(devnull, 'wb') as nullError:
        process = Popen([executable, script.path] + list(args), stdout=PIPE, stderr=nullError, stdin=nullInput)
        stdout = process.communicate()[0]
    return stdout