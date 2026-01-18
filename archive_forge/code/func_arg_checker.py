from datetime import datetime, timedelta
from unittest import TestCase
import macaroonbakery.checkers as checkers
import six
from pymacaroons import MACAROON_V2, Macaroon
def arg_checker(test, expect_cond, check_arg):
    """ Returns a checker function that checks that the caveat condition is
    check_arg.
    """

    def func(ctx, cond, arg):
        test.assertEqual(cond, expect_cond)
        if arg != check_arg:
            return 'wrong arg'
        return None
    return func