import copy
from oslo_utils import encodeutils
from oslo_utils import strutils
import urllib.parse
from glanceclient.common import utils
from glanceclient.v1.apiclient import base
def filter_owner(owner, image):
    if owner is None:
        return False
    if not hasattr(image, 'owner') or image.owner is None:
        return not owner == ''
    else:
        return not image.owner == owner