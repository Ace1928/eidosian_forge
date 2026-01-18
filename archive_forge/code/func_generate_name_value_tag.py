import six
def generate_name_value_tag(name, value):
    """Generate a CADF tag in the format name?value=<value>

    :param name: name of tag
    :param valuue: optional value tag
    """
    if name is None or value is None:
        raise ValueError('Invalid name and/or value. Values cannot be None')
    tag = name + '?value=' + value
    return tag