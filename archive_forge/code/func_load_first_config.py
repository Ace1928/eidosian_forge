import os, stat
def load_first_config(*resource):
    """Returns the first result from load_config_paths, or None if there is nothing
    to load."""
    for x in load_config_paths(*resource):
        return x
    return None