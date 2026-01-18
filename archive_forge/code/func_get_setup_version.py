import os, subprocess, json
@classmethod
def get_setup_version(cls, setup_path, reponame, describe=False, dirty='report', pkgname=None, archive_commit=None):
    """
        Helper for use in setup.py to get the version from the .version file (if available)
        or more up-to-date information from git describe (if available).

        Assumes the __init__.py will be found in the directory
        {reponame}/__init__.py relative to setup.py unless pkgname is
        explicitly specified in which case that name is used instead.

        If describe is True, the raw string obtained from git described is
        returned which is useful for updating the .version file.

        The dirty policy can be one of 'report', 'strip', 'raise'. If it is
        'report' the version string may end in '-dirty' if the repository is
        in a dirty state.  If the policy is 'strip', the '-dirty' suffix
        will be stripped out if present. If the policy is 'raise', an
        exception is raised if the repository is in a dirty state. This can
        be useful if you want to make sure packages are not built from a
        dirty repository state.
        """
    pkgname = reponame if pkgname is None else pkgname
    policies = ['raise', 'report', 'strip']
    if dirty not in policies:
        raise AssertionError('get_setup_version dirty policy must be in %r' % policies)
    fpath = os.path.join(setup_path, pkgname, '__init__.py')
    version = Version(fpath=fpath, reponame=reponame, archive_commit=archive_commit)
    if describe:
        vstring = version.git_fetch(as_string=True)
    else:
        vstring = str(version)
    if version.dirty and dirty == 'raise':
        raise AssertionError('Repository is in a dirty state.')
    elif version.dirty and dirty == 'strip':
        return vstring.replace('-dirty', '')
    else:
        return vstring