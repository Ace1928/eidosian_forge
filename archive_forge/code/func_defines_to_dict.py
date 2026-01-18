def defines_to_dict(defines):
    """Convert a list of definition strings to a dictionary."""
    if defines is None:
        return None
    result = {}
    for define in defines:
        kv = define.split('=', 1)
        if len(kv) == 1:
            result[define.strip()] = 1
        else:
            result[kv[0].strip()] = kv[1].strip()
    return result