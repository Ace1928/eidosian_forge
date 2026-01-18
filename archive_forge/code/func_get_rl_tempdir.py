import os, tempfile
def get_rl_tempdir(*subdirs):
    global _rl_tempdir
    if _rl_tempdir is None:
        _rl_tempdir = os.path.join(tempfile.gettempdir(), 'ReportLab_tmp%s' % str(_rl_getuid()))
    d = _rl_tempdir
    if subdirs:
        d = os.path.join(*(d,) + subdirs)
    try:
        os.makedirs(d)
    except:
        pass
    return d