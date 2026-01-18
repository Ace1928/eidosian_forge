import os, stat
def load_config_paths(*resource):
    """Returns an iterator which gives each directory named 'resource' in the
    configuration search path. Information provided by earlier directories should
    take precedence over later ones, and the user-specific config dir comes
    first."""
    resource = os.path.join(*resource)
    for config_dir in xdg_config_dirs:
        path = os.path.join(config_dir, resource)
        if os.path.exists(path):
            yield path