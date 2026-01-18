import reportlab
import sys, os, fnmatch, re, functools
from configparser import ConfigParser
import unittest
from reportlab.lib.utils import isCompactDistro, __rl_loader__, rl_isdir, asUnicode
def getCVSEntries(folder, files=1, folders=0):
    """Returns a list of filenames as listed in the CVS/Entries file.

    'folder' is the folder that should contain the CVS subfolder.
    If there is no such subfolder an empty list is returned.
    'files' is a boolean; 1 and 0 means to return files or not.
    'folders' is a boolean; 1 and 0 means to return folders or not.
    """
    join = os.path.join
    try:
        f = open(join(folder, 'CVS', 'Entries'))
    except IOError:
        return []
    allEntries = []
    for line in f.readlines():
        if folders and line[0] == 'D' or (files and line[0] != 'D'):
            entry = line.split('/')[1]
            if entry:
                allEntries.append(join(folder, entry))
    return allEntries