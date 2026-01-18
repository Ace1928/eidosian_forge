import sys
import re
import os
from configparser import RawConfigParser
def parse_meta(config):
    if not config.has_section('meta'):
        raise FormatError('No meta section found !')
    d = dict(config.items('meta'))
    for k in ['name', 'description', 'version']:
        if not k in d:
            raise FormatError('Option %s (section [meta]) is mandatory, but not found' % k)
    if not 'requires' in d:
        d['requires'] = []
    return d