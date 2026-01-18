import copy
import functools
import os
import sys
from oslo_utils import encodeutils
from oslo_utils import strutils
from glanceclient.common import progressbar
from glanceclient.common import utils
from glanceclient import exc
import glanceclient.v1.images
def convert_size(image):
    image.size = utils.make_size_human_readable(image.size)
    return image