import uuid
from osc_placement.tests.functional import base
def sorted_resources(resource):
    return ','.join(sorted(resource.split(',')))