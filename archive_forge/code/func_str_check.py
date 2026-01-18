from datetime import datetime, timedelta
import macaroonbakery.bakery as bakery
import macaroonbakery.checkers as checkers
def str_check(ctx, cond, args):
    expect = ctx[_str_key]
    if args != expect:
        return "{} doesn't match {}".format(cond, expect)
    return None