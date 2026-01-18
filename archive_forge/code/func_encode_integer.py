import sys
import os
import boto
import optparse
import copy
import boto.exception
import boto.roboto.awsqueryservice
import bdb
import traceback
@classmethod
def encode_integer(cls, p, rp, v, l):
    if l:
        label = l
    else:
        label = p.name
    rp[label] = '%d' % v