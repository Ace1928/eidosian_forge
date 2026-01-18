def standardize_padding(value, allow_causal=False):
    if isinstance(value, (list, tuple)):
        return value
    padding = value.lower()
    if allow_causal:
        allowed_values = {'valid', 'same', 'causal'}
    else:
        allowed_values = {'valid', 'same'}
    if padding not in allowed_values:
        raise ValueError(f'The `padding` argument must be a list/tuple or one of {allowed_values}. Received: {padding}')
    return padding