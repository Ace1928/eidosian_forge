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
def senderror(failure, options):
    recipient = [options.sender]
    sender = '"Internally Generated Message ({})"<postmaster@{}>'.format(sys.argv[0], smtp.DNSNAME.decode('ascii'))
    error = StringIO()
    failure.printTraceback(file=error)
    body = StringIO(ERROR_FMT % error.getvalue())
    d = smtp.sendmail('localhost', sender, recipient, body)
    d.addBoth(lambda _: reactor.stop())