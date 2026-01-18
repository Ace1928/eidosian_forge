import functools
from webob import exc
from heat.common.i18n import _
from heat.common import identifier
def get_allowed_params(params, param_types):
    """Extract from ``params`` all entries listed in ``param_types``.

    The returning dict will contain an entry for a key if, and only if,
    there's an entry in ``param_types`` for that key and at least one entry in
    ``params``. If ``params`` contains multiple entries for the same key, it
    will yield an array of values: ``{key: [v1, v2,...]}``

    :param params: a NestedMultiDict from webob.Request.params
    :param param_types: an dict of allowed parameters and their types

    :returns: a dict with {key: value} pairs
    """
    allowed_params = {}
    for key, get_type in param_types.items():
        assert get_type in PARAM_TYPES
        value = None
        if get_type == PARAM_TYPE_SINGLE:
            value = params.get(key)
        elif get_type == PARAM_TYPE_MULTI:
            value = params.getall(key)
        elif get_type == PARAM_TYPE_MIXED:
            value = params.getall(key)
            if isinstance(value, list) and len(value) == 1:
                value = value.pop()
        if value:
            allowed_params[key] = value
    return allowed_params