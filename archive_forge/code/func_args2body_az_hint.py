from neutronclient._i18n import _
from neutronclient.neutron import v2_0 as neutronv20
def args2body_az_hint(parsed_args, resource):
    if parsed_args.availability_zone_hints:
        resource['availability_zone_hints'] = parsed_args.availability_zone_hints