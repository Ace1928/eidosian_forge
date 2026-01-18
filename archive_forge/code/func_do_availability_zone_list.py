from zunclient.common import utils as zun_utils
def do_availability_zone_list(cs, args):
    """Print a list of availability zones."""
    zones = cs.availability_zones.list()
    zun_utils.list_availability_zones(zones)