import base64
import hashlib
import itertools
import os
import google
from ._checker import (Op, LOGIN_OP)
from ._store import MemoryKeyStore
from ._error import VerificationError
from ._versions import (
from ._macaroon import (
import macaroonbakery.checkers as checkers
import six
from macaroonbakery._utils import (
from ._internal import id_pb2
from pymacaroons import MACAROON_V2, Verifier
def ops_entity(self, ops):
    """ Returns a new multi-op entity name string that represents
        all the given operations and caveats. It returns the same value
        regardless of the ordering of the operations. It assumes that the
        operations have been canonicalized and that there's at least one
        operation.

        :param ops:
        :return: string that represents all the given operations and caveats.
        """
    hash_entity = hashlib.sha256()
    for op in ops:
        hash_entity.update('{}\n{}\n'.format(op.action, op.entity).encode())
    hash_encoded = base64.urlsafe_b64encode(hash_entity.digest())
    return 'multi-' + hash_encoded.decode('utf-8').rstrip('=')