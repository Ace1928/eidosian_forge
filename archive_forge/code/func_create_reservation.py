import logging
from typing import List, Optional, Union
from google.api_core.exceptions import InvalidArgument
from google.api_core.operation import Operation
from cloudsdk.google.protobuf.field_mask_pb2 import FieldMask  # pytype: disable=pyi-error
from google.cloud.pubsublite.admin_client_interface import AdminClientInterface
from google.cloud.pubsublite.types import (
from google.cloud.pubsublite.types.paths import ReservationPath
from google.cloud.pubsublite_v1 import (
def create_reservation(self, reservation: Reservation) -> Reservation:
    path = ReservationPath.parse(reservation.name)
    return self._underlying.create_reservation(parent=str(path.to_location_path()), reservation=reservation, reservation_id=path.name)