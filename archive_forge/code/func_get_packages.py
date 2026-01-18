import os
import fnmatch
def get_packages(dname, pkgname=None, results=None, ignore=None, parent=None):
    """
    Get all packages which are under dname. This is necessary for
    Python 2.2's distutils. Pretty similar arguments to getDataFiles,
    including 'parent'.
    """
    parent = parent or ''
    prefix = []
    if parent:
        prefix = [parent]
    bname = os.path.basename(dname)
    ignore = ignore or []
    if bname in ignore:
        return []
    if results is None:
        results = []
    if pkgname is None:
        pkgname = []
    subfiles = os.listdir(dname)
    abssubfiles = [os.path.join(dname, x) for x in subfiles]
    if '__init__.py' in subfiles:
        results.append(prefix + pkgname + [bname])
        for subdir in filter(os.path.isdir, abssubfiles):
            get_packages(subdir, pkgname=pkgname + [bname], results=results, ignore=ignore, parent=parent)
    res = ['.'.join(result) for result in results]
    return res