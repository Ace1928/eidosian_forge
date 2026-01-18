import os
import re
import stat
import sys
import fnmatch
from xdg import BaseDirectory
import xdg.Locale
from xml.dom import minidom, XML_NAMESPACE
from collections import defaultdict
def install_mime_info(application, package_file):
    """Copy 'package_file' as ``~/.local/share/mime/packages/<application>.xml.``
    If package_file is None, install ``<app_dir>/<application>.xml``.
    If already installed, does nothing. May overwrite an existing
    file with the same name (if the contents are different)"""
    application += '.xml'
    with open(package_file) as f:
        new_data = f.read()
    package_dir = os.path.join('mime', 'packages')
    resource = os.path.join(package_dir, application)
    for x in BaseDirectory.load_data_paths(resource):
        try:
            with open(x) as f:
                old_data = f.read()
        except:
            continue
        if old_data == new_data:
            return
    global _cache_uptodate
    _cache_uptodate = False
    new_file = os.path.join(BaseDirectory.save_data_path(package_dir), application)
    with open(new_file, 'w') as f:
        f.write(new_data)
    command = 'update-mime-database'
    if os.spawnlp(os.P_WAIT, command, command, BaseDirectory.save_data_path('mime')):
        os.unlink(new_file)
        raise Exception("The '%s' command returned an error code!\nMake sure you have the freedesktop.org shared MIME package:\nhttp://standards.freedesktop.org/shared-mime-info/" % command)