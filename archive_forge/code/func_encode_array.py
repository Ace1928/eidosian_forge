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
def encode_array(cls, p, rp, v, l):
    v = boto.utils.mklist(v)
    if l:
        label = l
    else:
        label = p.name
    label = label + '.%d'
    for i, value in enumerate(v):
        rp[label % (i + 1)] = value