from designateclient import client
from designateclient import exceptions
from heat.common import exception as heat_exception
from heat.engine.clients import client_plugin
from heat.engine import constraints
def get_zone_id(self, zone_id_or_name):
    client = self.client()
    try:
        zone_obj = client.zones.get(zone_id_or_name)
        return zone_obj['id']
    except exceptions.NotFound:
        zones = client.zones.list(criterion=dict(name=zone_id_or_name))
        if len(zones) == 1:
            return zones[0]['id']
    raise heat_exception.EntityNotFound(entity='Designate Zone', name=zone_id_or_name)