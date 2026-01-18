import os, tempfile
def get_rl_tempfile(fn=None):
    if not fn:
        fn = tempfile.mktemp()
    return os.path.join(get_rl_tempdir(), fn)