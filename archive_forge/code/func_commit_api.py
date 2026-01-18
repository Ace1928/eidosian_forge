import sys
from functools import partial
from pydev_ipython.version import check_version
def commit_api(api):
    """Commit to a particular API, and trigger ImportErrors on subsequent
       dangerous imports"""
    if api == QT_API_PYSIDE:
        ID.forbid('PyQt4')
        ID.forbid('PyQt5')
    else:
        ID.forbid('PySide')
        ID.forbid('PySide2')