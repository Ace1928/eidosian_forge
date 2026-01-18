def get_nipype_gitversion():
    """Nipype version as reported by the last commit in git

    Returns
    -------
    None or str
      Version of Nipype according to git.
    """
    import os
    import subprocess
    try:
        import nipype
        gitpath = os.path.realpath(os.path.join(os.path.dirname(nipype.__file__), os.path.pardir))
    except:
        gitpath = os.getcwd()
    gitpathgit = os.path.join(gitpath, '.git')
    if not os.path.exists(gitpathgit):
        return None
    ver = None
    try:
        o, _ = subprocess.Popen('git describe', shell=True, cwd=gitpath, stdout=subprocess.PIPE).communicate()
    except Exception:
        pass
    else:
        ver = o.decode().strip().split('-')[-1]
    return ver