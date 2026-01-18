import datetime
from redis.utils import str_if_bytes
def parse_info(response):
    """Parse the result of Redis's INFO command into a Python dict"""
    info = {}
    response = str_if_bytes(response)

    def get_value(value):
        if ',' not in value or '=' not in value:
            try:
                if '.' in value:
                    return float(value)
                else:
                    return int(value)
            except ValueError:
                return value
        else:
            sub_dict = {}
            for item in value.split(','):
                k, v = item.rsplit('=', 1)
                sub_dict[k] = get_value(v)
            return sub_dict
    for line in response.splitlines():
        if line and (not line.startswith('#')):
            if line.find(':') != -1:
                key, value = line.split(':', 1)
                if key == 'cmdstat_host':
                    key, value = line.rsplit(':', 1)
                if key == 'module':
                    info.setdefault('modules', []).append(get_value(value))
                else:
                    info[key] = get_value(value)
            else:
                info.setdefault('__raw__', []).append(line)
    return info