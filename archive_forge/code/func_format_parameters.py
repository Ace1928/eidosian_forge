import numbers
import prettytable
import yaml
from osc_lib import exceptions as exc
from oslo_serialization import jsonutils
def format_parameters(params):
    """Reformat parameters into dict of format expected by the API."""
    if not params:
        return {}
    if len(params) == 1:
        if params[0].find(';') != -1:
            params = params[0].split(';')
        else:
            params = params[0].split(',')
    parameters = {}
    for p in params:
        try:
            n, v = p.split('=', 1)
        except ValueError:
            msg = '%s(%s). %s.' % ('Malformed parameter', p, 'Use the key=value format')
            raise exc.CommandError(msg)
        if n not in parameters:
            parameters[n] = v
        else:
            if not isinstance(parameters[n], list):
                parameters[n] = [parameters[n]]
            parameters[n].append(v)
    return parameters