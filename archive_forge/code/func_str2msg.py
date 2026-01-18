from .specs import SPEC_BY_TYPE, make_msgdict
def str2msg(text):
    """Parse str format and return message dict.

    No type or value checking is done. The caller is responsible for
    calling check_msgdict().
    """
    words = text.split()
    type_ = words[0]
    args = words[1:]
    msg = {}
    for arg in args:
        name, value = arg.split('=', 1)
        if name == 'time':
            value = _parse_time(value)
        elif name == 'data':
            value = _parse_data(value)
        else:
            value = int(value)
        msg[name] = value
    return make_msgdict(type_, msg)