import os
def make_dirs_if_not_exists(name):
    try:
        os.makedirs(name)
    except OSError:
        pass