import sys
import os
import boto
import optparse
import copy
import boto.exception
import boto.roboto.awsqueryservice
import bdb
import traceback
@property
def request_id(self):
    retval = None
    if self.aws_response is not None:
        retval = getattr(self.aws_response, 'requestId')
    return retval