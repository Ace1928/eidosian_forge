import os, stat
def save_state_path(*resource):
    """Ensure ``$XDG_STATE_HOME/<resource>/`` exists, and return its path.
    'resource' should normally be the name of your application or a shared
    resource."""
    resource = os.path.join(*resource)
    assert not resource.startswith('/')
    path = os.path.join(xdg_state_home, resource)
    if not os.path.isdir(path):
        os.makedirs(path)
    return path