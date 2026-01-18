import sys
def getDOMImplementation(name=None, features=()):
    """getDOMImplementation(name = None, features = ()) -> DOM implementation.

    Return a suitable DOM implementation. The name is either
    well-known, the module name of a DOM implementation, or None. If
    it is not None, imports the corresponding module and returns
    DOMImplementation object if the import succeeds.

    If name is not given, consider the available implementations to
    find one with the required feature set. If no implementation can
    be found, raise an ImportError. The features list must be a sequence
    of (feature, version) pairs which are passed to hasFeature."""
    import os
    creator = None
    mod = well_known_implementations.get(name)
    if mod:
        mod = __import__(mod, {}, {}, ['getDOMImplementation'])
        return mod.getDOMImplementation()
    elif name:
        return registered[name]()
    elif not sys.flags.ignore_environment and 'PYTHON_DOM' in os.environ:
        return getDOMImplementation(name=os.environ['PYTHON_DOM'])
    if isinstance(features, str):
        features = _parse_feature_string(features)
    for creator in registered.values():
        dom = creator()
        if _good_enough(dom, features):
            return dom
    for creator in well_known_implementations.keys():
        try:
            dom = getDOMImplementation(name=creator)
        except Exception:
            continue
        if _good_enough(dom, features):
            return dom
    raise ImportError('no suitable DOM implementation found')