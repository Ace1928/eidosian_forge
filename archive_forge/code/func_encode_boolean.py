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
def encode_boolean(cls, p, rp, v, l):
    if l:
        label = l
    else:
        label = p.name
    if v:
        v = 'true'
    else:
        v = 'false'
    rp[label] = v