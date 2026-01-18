import email.utils
import getpass
import os
import sys
from configparser import ConfigParser
from io import StringIO
from twisted.copyright import version
from twisted.internet import reactor
from twisted.logger import Logger, textFileLogObserver
from twisted.mail import smtp
def failure(f):
    global failed
    reactor.stop()
    failed = f