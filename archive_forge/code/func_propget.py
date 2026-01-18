import os, sys, time, re, calendar
import py
import subprocess
from py._path import common
def propget(self, name):
    """ get property name on this path. """
    res = self._svn('propget', name)
    return res[:-1]