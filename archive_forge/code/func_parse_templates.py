import os.path
from oslo_log import log
from keystone.catalog.backends import base
from keystone.common import utils
import keystone.conf
from keystone import exception
def parse_templates(template_lines):
    o = {}
    for line in template_lines:
        if ' = ' not in line:
            continue
        k, v = line.strip().split(' = ')
        if not k.startswith('catalog.'):
            continue
        parts = k.split('.')
        region = parts[1]
        service = parts[2].replace('_', '-')
        key = parts[3]
        region_ref = o.get(region, {})
        service_ref = region_ref.get(service, {})
        service_ref[key] = v
        region_ref[service] = service_ref
        o[region] = region_ref
    return o