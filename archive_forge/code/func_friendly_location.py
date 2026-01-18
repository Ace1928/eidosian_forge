import shutil
import sys
from breezy import (branch, controldir, errors, info, osutils, tests, upgrade,
from breezy.bzr import bzrdir
from breezy.transport import memory
def friendly_location(url):
    path = urlutils.unescape_for_display(url, 'ascii')
    try:
        return osutils.relpath(osutils.getcwd(), path)
    except errors.PathNotChild:
        return path