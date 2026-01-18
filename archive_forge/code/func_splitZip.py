import os
import sys
import zipfile
def splitZip(path):
    """Splits a path containing a zip file into (zipfile, subpath).
    If there is no zip file, returns (path, None)"""
    components = os.path.normpath(path).split(os.sep)
    for index, component in enumerate(components):
        if component.endswith('.zip'):
            zipPath = os.sep.join(components[0:index + 1])
            archivePath = ''.join([x + '/' for x in components[index + 1:]])
            return (zipPath, archivePath)
    else:
        return (path, None)