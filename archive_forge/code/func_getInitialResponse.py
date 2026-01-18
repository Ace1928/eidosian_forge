import binascii
import os
import random
import time
from hashlib import md5
from zope.interface import Attribute, Interface, implementer
from twisted.python.compat import networkString
def getInitialResponse(self):
    return None