from aiokeydb.v1.commands.helpers import nativestr
def parse_m_range(response):
    """Parse multi range response. Used by TS.MRANGE and TS.MREVRANGE."""
    res = []
    for item in response:
        res.append({nativestr(item[0]): [list_to_dict(item[1]), parse_range(item[2])]})
    return sorted(res, key=lambda d: list(d.keys()))