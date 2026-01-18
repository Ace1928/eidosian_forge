import io, logging, socket, os, pickle, struct, time, re
from stat import ST_DEV, ST_INO, ST_MTIME
import queue
import threading
import copy
def getFilesToDelete(self):
    """
        Determine the files to delete when rolling over.

        More specific than the earlier method, which just used glob.glob().
        """
    dirName, baseName = os.path.split(self.baseFilename)
    fileNames = os.listdir(dirName)
    result = []
    n, e = os.path.splitext(baseName)
    prefix = n + '.'
    plen = len(prefix)
    for fileName in fileNames:
        if self.namer is None:
            if not fileName.startswith(baseName):
                continue
        elif not fileName.startswith(baseName) and fileName.endswith(e) and (len(fileName) > plen + 1) and (not fileName[plen + 1].isdigit()):
            continue
        if fileName[:plen] == prefix:
            suffix = fileName[plen:]
            parts = suffix.split('.')
            for part in parts:
                if self.extMatch.match(part):
                    result.append(os.path.join(dirName, fileName))
                    break
    if len(result) < self.backupCount:
        result = []
    else:
        result.sort()
        result = result[:len(result) - self.backupCount]
    return result