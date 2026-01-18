def dict_resolver(attname, default_value, root, info, **args):
    return root.get(attname, default_value)