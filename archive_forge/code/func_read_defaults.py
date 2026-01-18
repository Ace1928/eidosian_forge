import runpy
def read_defaults():
    import os
    name = os.path.expanduser('~/.ase/gui.py')
    config = gui_default_settings
    if os.path.exists(name):
        runpy.run_path(name, init_globals={'gui_default_settings': gui_default_settings})
    return config