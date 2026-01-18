from ._auth_context import ContextKey
from ._caveat import Caveat, error_caveat, parse_caveat
from ._conditions import (
from ._namespace import Namespace
def infer_declared(ms, namespace=None):
    """Retrieves any declared information from the given macaroons and returns
    it as a key-value map.
    Information is declared with a first party caveat as created by
    declared_caveat.

    If there are two caveats that declare the same key with different values,
    the information is omitted from the map. When the caveats are later
    checked, this will cause the check to fail.
    namespace is the Namespace used to retrieve the prefix associated to the
    uri, if None it will use the STD_NAMESPACE only.
    """
    conditions = []
    for m in ms:
        for cav in m.caveats:
            if cav.location is None or cav.location == '':
                conditions.append(cav.caveat_id_bytes.decode('utf-8'))
    return infer_declared_from_conditions(conditions, namespace)