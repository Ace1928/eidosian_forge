import os
import shutil
import stat
import tarfile
import zipfile
from django.core.exceptions import SuspiciousOperation
def split_leading_dir(self, path):
    path = str(path)
    path = path.lstrip('/').lstrip('\\')
    if '/' in path and ('\\' in path and path.find('/') < path.find('\\') or '\\' not in path):
        return path.split('/', 1)
    elif '\\' in path:
        return path.split('\\', 1)
    else:
        return (path, '')