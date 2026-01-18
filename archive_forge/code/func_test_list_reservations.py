import datetime
import pytest
from google.protobuf.timestamp_pb2 import Timestamp
from cirq_google.cloud import quantum
from cirq_google.engine.abstract_local_processor import AbstractLocalProcessor
def test_list_reservations():
    p = NothingProcessor(processor_id='test')
    now = datetime.datetime.now()
    hour = datetime.timedelta(hours=1)
    users = ['abc@def.com']
    reservation1 = p.create_reservation(start_time=now - hour, end_time=now, whitelisted_users=users)
    reservation2 = p.create_reservation(start_time=now, end_time=now + hour, whitelisted_users=users)
    reservation3 = p.create_reservation(start_time=now + hour, end_time=now + 2 * hour, whitelisted_users=users)
    assert p.list_reservations(now - 2 * hour, now + 3 * hour) == [reservation1, reservation2, reservation3]
    assert p.list_reservations(now + 0.5 * hour, now + 3 * hour) == [reservation2, reservation3]
    assert p.list_reservations(now + 1.5 * hour, now + 3 * hour) == [reservation3]
    assert p.list_reservations(now + 0.5 * hour, now + 0.75 * hour) == [reservation2]
    assert p.list_reservations(now - 1.5 * hour, now + 0.5 * hour) == [reservation1, reservation2]
    assert p.list_reservations(0.5 * hour, 3 * hour) == [reservation2, reservation3]
    assert p.list_reservations(1.5 * hour, 3 * hour) == [reservation3]
    assert p.list_reservations(0.25 * hour, 0.5 * hour) == [reservation2]
    assert p.list_reservations(-1.5 * hour, 0.5 * hour) == [reservation1, reservation2]
    assert p.list_reservations(now - 2 * hour, None) == [reservation1, reservation2, reservation3]
    p.remove_reservation(reservation1.name)
    assert p.list_reservations(now - 2 * hour, None) == [reservation2, reservation3]