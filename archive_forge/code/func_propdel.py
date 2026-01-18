import os, sys, time, re, calendar
import py
import subprocess
from py._path import common
def propdel(self, name):
    """ delete property name on this path. """
    res = self._svn('propdel', name)
    return res[:-1]