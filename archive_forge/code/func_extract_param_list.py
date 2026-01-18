import itertools
import re
from oslo_log import log as logging
from heat.api.aws import exception
def extract_param_list(params, prefix=''):
    """Extract a list-of-dicts based on parameters containing AWS style list.

    MetricData.member.1.MetricName=buffers
    MetricData.member.1.Unit=Bytes
    MetricData.member.1.Value=231434333
    MetricData.member.2.MetricName=buffers2
    MetricData.member.2.Unit=Bytes
    MetricData.member.2.Value=12345

    This can be extracted by passing prefix=MetricData, resulting in a
    list containing two dicts.
    """
    key_re = re.compile('%s\\.member\\.([0-9]+)\\.(.*)' % prefix)

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

    def key_func(d):
        return d[0]
    data = sorted(get_param_data(params), key=key_func)
    members = itertools.groupby(data, key_func)
    return [dict((kv for di, kv in m)) for mi, m in members]