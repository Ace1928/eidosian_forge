import contextlib
import io
import os
import sys
import warnings
def getuser():
    """Get the username from the environment or password database.

    First try various environment variables, then the password
    database.  This works on Windows as long as USERNAME is set.

    """
    for name in ('LOGNAME', 'USER', 'LNAME', 'USERNAME'):
        user = os.environ.get(name)
        if user:
            return user
    import pwd
    return pwd.getpwuid(os.getuid())[0]