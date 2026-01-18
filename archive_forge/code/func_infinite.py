import itertools
import unittest
import sys
from autopage.tests import isolation
import typing
import autopage
def infinite() -> int:
    ap = autopage.AutoPager(pager_command=autopage.command.Less())
    try:
        with ap as out:
            for i in itertools.count():
                print(i, file=out)
    except KeyboardInterrupt:
        pass
    return ap.exit_code()