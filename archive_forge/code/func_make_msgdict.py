def make_msgdict(type_, overrides):
    """Return a new message.

    Returns a dictionary representing a message.

    Message values can be overriden.

    No type or value checking is done.  The caller is responsible for
    calling check_msgdict().
    """
    if type_ in SPEC_BY_TYPE:
        spec = SPEC_BY_TYPE[type_]
    else:
        raise LookupError(f'Unknown message type {type_!r}')
    msg = {'type': type_, 'time': DEFAULT_VALUES['time']}
    for name in spec['value_names']:
        msg[name] = DEFAULT_VALUES[name]
    msg.update(overrides)
    return msg