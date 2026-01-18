from ._authorizer import ClosedAuthorizer
from ._checker import Checker
import macaroonbakery.checkers as checkers
from ._oven import Oven
def root_keystore_for_ops(ops):
    return root_key_store