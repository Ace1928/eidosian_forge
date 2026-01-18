def remove_extension(module, name, code):
    """Unregister an extension code.  For testing only."""
    key = (module, name)
    if _extension_registry.get(key) != code or _inverted_registry.get(code) != key:
        raise ValueError('key %s is not registered with code %s' % (key, code))
    del _extension_registry[key]
    del _inverted_registry[code]
    if code in _extension_cache:
        del _extension_cache[code]