def dict_or_attr_resolver(attname, default_value, root, info, **args):
    resolver = dict_resolver if isinstance(root, dict) else attr_resolver
    return resolver(attname, default_value, root, info, **args)