import logging
import sys
import traceback
import cookielib
from urlparse import parse_qs
from saml2test import CheckError
from saml2test import FatalError
from saml2test import OperationError
from saml2test.check import ERROR
from saml2test.check import ExpectedError
from saml2test.interaction import Action
from saml2test.interaction import Interaction
from saml2test.interaction import InteractionNeeded
from saml2test.opfunc import Operation
from saml2test.status import INTERACTION
from saml2test.status import STATUSCODE
def check_severity(self, stat):
    if stat['status'] >= 4:
        logger.error('WHERE: %s', stat['id'])
        logger.error('STATUS:%s', STATUSCODE[stat['status']])
        try:
            logger.error('HTTP STATUS: %s', stat['http_status'])
        except KeyError:
            pass
        try:
            logger.error('INFO: %s', stat['message'])
        except KeyError:
            pass
        raise CheckError