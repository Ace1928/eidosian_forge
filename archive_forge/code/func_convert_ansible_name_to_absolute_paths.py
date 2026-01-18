from __future__ import (absolute_import, division, print_function)
def convert_ansible_name_to_absolute_paths(name):
    """Calculate the module path from the given name.
        :type name: str
        :rtype: list[str]
        """
    return [os.path.join(ansible_path, name.replace('.', os.path.sep)), os.path.join(ansible_path, name.replace('.', os.path.sep)) + '.py']