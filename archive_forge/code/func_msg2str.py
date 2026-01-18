from .specs import SPEC_BY_TYPE, make_msgdict
def msg2str(msg, include_time=True):
    type_ = msg['type']
    spec = SPEC_BY_TYPE[type_]
    words = [type_]
    for name in spec['value_names']:
        value = msg[name]
        if name == 'data':
            value = '({})'.format(','.join((str(byte) for byte in value)))
        words.append(f'{name}={value}')
    if include_time:
        words.append('time={}'.format(msg['time']))
    return str.join(' ', words)