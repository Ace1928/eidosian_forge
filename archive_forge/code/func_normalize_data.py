from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
def normalize_data(cmd_ref):
    """Normalize playbook values and get_existing data"""
    playval = cmd_ref._ref.get('destination').get('playval')
    existing = cmd_ref._ref.get('destination').get('existing')
    dest_props = ['protocol', 'encoding']
    if playval:
        for prop in dest_props:
            for key in playval.keys():
                playval[key][prop] = playval[key][prop].lower()
    if existing:
        for key in existing.keys():
            for prop in dest_props:
                existing[key][prop] = existing[key][prop].lower()