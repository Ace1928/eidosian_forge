from __future__ import (absolute_import, division, print_function)
def is_name_in_namepace(name, namespaces):
    """Returns True if the given name is one of the given namespaces, otherwise returns False."""
    name_parts = name.split('.')
    for namespace in namespaces:
        namespace_parts = namespace.split('.')
        length = min(len(name_parts), len(namespace_parts))
        truncated_name = name_parts[0:length]
        truncated_namespace = namespace_parts[0:length]
        for idx, part in enumerate(truncated_namespace):
            if not part:
                truncated_name[idx] = part
        if truncated_name == truncated_namespace:
            return True
    return False