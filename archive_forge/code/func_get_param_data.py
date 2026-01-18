import itertools
import re
from oslo_log import log as logging
from heat.api.aws import exception
def get_param_data(params):
    for param_name, value in params.items():
        match = key_re.match(param_name)
        if match:
            try:
                index = int(match.group(1))
            except ValueError:
                pass
            else:
                key = match.group(2)
                yield (index, (key, value))