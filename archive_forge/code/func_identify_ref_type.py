import logging
from oslo_utils.timeutils import parse_isotime
from barbicanclient import base
from barbicanclient import formatter
@staticmethod
def identify_ref_type(entity_ref):
    if not entity_ref:
        raise ValueError('Secret or container href is required.')
    if '/secrets' in entity_ref:
        ref_type = 'secret'
    elif '/containers' in entity_ref:
        ref_type = 'container'
    else:
        raise ValueError('Secret or container URI is not specified.')
    return ref_type