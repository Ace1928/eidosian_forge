from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import time
def specific_reservation_update(module, request, response):
    auth = GcpSession(module, 'compute')
    auth.post(''.join(['https://compute.googleapis.com/compute/v1/', 'projects/{project}/zones/{zone}/reservations/{name}/resize']).format(**module.params), {u'specificReservation': ReservationSpecificreservation(module.params.get('specific_reservation', {}), module).to_request()})