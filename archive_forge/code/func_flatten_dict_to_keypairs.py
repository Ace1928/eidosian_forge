def flatten_dict_to_keypairs(d, separator=':'):
    """Generator that produces sequence of keypairs for nested dictionaries.

    :param d: dictionaries which may be nested
    :param separator: symbol between names
    """
    for name, value in sorted(d.items()):
        if isinstance(value, dict):
            for subname, subvalue in flatten_dict_to_keypairs(value, separator):
                yield ('%s%s%s' % (name, separator, subname), subvalue)
        else:
            yield (name, value)