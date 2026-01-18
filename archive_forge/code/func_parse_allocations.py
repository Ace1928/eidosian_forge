from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from osc_placement import version
def parse_allocations(allocation_strings):
    allocations = {}
    for allocation_string in allocation_strings:
        if '=' not in allocation_string or ',' not in allocation_string:
            raise ValueError('Incorrect allocation string format')
        parsed = dict((kv.split('=') for kv in allocation_string.split(',')))
        if 'rp' not in parsed:
            raise ValueError('Resource provider parameter is required for allocation string')
        resources = {k: int(v) for k, v in parsed.items() if k != 'rp'}
        if parsed['rp'] not in allocations:
            allocations[parsed['rp']] = resources
        else:
            prev_rp = allocations[parsed['rp']]
            for resource, value in resources.items():
                if resource in prev_rp and prev_rp[resource] != value:
                    raise exceptions.CommandError('Conflict detected for resource provider {} resource class {}'.format(parsed['rp'], resource))
            allocations[parsed['rp']].update(resources)
    return allocations