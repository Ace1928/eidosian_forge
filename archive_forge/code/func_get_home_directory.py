import os
def get_home_directory():
    """ Determine the user's home directory."""
    for name in ['HOME', 'USERPROFILE']:
        if name in os.environ:
            path = os.environ[name]
            if path[-1] != os.path.sep:
                path += os.path.sep
            break
    else:
        path = ''
    return path