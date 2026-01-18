import os
from operator import getitem
from twisted.python.compat import _PYPY
from twisted.python.fakepwd import ShadowDatabase, UserDatabase
from twisted.trial.unittest import TestCase
def findInvalidUID():
    """
    By convention, UIDs less than 1000 are reserved for the system.  A system
    which allocated every single one of those UIDs would likely have practical
    problems with allocating new ones, so let's assume that we'll be able to
    find one.  (If we don't, this will wrap around to negative values and
    I{eventually} find something.)

    @return: a user ID which does not exist on the local system.  Or, on
        systems without a L{pwd} module, return C{SYSTEM_UID_MAX}.
    """
    guess = SYSTEM_UID_MAX
    if pwd is not None:
        while True:
            try:
                pwd.getpwuid(guess)
            except KeyError:
                break
            else:
                guess -= 1
    return guess