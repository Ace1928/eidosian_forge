def set_debug_flags_from_config():
    """Turn on debug flags based on the global configuration"""
    from breezy import config
    c = config.GlobalStack()
    for f in c.get('debug_flags'):
        debug_flags.add(f)