from aiokeydb.v1.commands.helpers import nativestr
def parse_get(response):
    """Parse get response. Used by TS.GET."""
    if not response:
        return None
    return (int(response[0]), float(response[1]))