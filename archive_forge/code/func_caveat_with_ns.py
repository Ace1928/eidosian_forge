from datetime import datetime, timedelta
from unittest import TestCase
import macaroonbakery.checkers as checkers
import six
from pymacaroons import MACAROON_V2, Macaroon
def caveat_with_ns(cav, ns):
    return checkers.Caveat(location=cav.location, condition=cav.condition, namespace=ns)