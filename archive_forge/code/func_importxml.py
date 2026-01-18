import os, sys, time, re, calendar
import py
import subprocess
from py._path import common
def importxml(cache=[]):
    if cache:
        return cache
    from xml.dom import minidom
    from xml.parsers.expat import ExpatError
    cache.extend([minidom, ExpatError])
    return cache