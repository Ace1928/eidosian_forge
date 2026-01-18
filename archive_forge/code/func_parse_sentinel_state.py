import datetime
from redis.utils import str_if_bytes
def parse_sentinel_state(item):
    result = pairs_to_dict_typed(item, SENTINEL_STATE_TYPES)
    flags = set(result['flags'].split(','))
    for name, flag in (('is_master', 'master'), ('is_slave', 'slave'), ('is_sdown', 's_down'), ('is_odown', 'o_down'), ('is_sentinel', 'sentinel'), ('is_disconnected', 'disconnected'), ('is_master_down', 'master_down')):
        result[name] = flag in flags
    return result