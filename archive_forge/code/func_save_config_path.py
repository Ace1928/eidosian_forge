import os, stat
def save_config_path(*resource):
    """Ensure ``$XDG_CONFIG_HOME/<resource>/`` exists, and return its path.
    'resource' should normally be the name of your application. Use this
    when saving configuration settings.
    """
    resource = os.path.join(*resource)
    assert not resource.startswith('/')
    path = os.path.join(xdg_config_home, resource)
    if not os.path.isdir(path):
        os.makedirs(path, 448)
    return path