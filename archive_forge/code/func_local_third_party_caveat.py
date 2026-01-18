import abc
from collections import namedtuple
from ._error import (
from ._codec import decode_caveat
from ._macaroon import (
from ._versions import VERSION_2
from ._third_party import ThirdPartyCaveatInfo
import macaroonbakery.checkers as checkers
def local_third_party_caveat(key, version):
    """ Returns a third-party caveat that, when added to a macaroon with
    add_caveat, results in a caveat with the location "local", encrypted with
    the given PublicKey.
    This can be automatically discharged by discharge_all passing a local key.
    """
    if version >= VERSION_2:
        loc = 'local {} {}'.format(version, key)
    else:
        loc = 'local {}'.format(key)
    return checkers.Caveat(location=loc, condition='')