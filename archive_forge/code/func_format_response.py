import itertools
import re
from oslo_log import log as logging
from heat.api.aws import exception
def format_response(action, response):
    """Format response from engine into API format."""
    return {'%sResponse' % action: {'%sResult' % action: response}}