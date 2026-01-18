import pytest
def test_devices():
    """test device imports"""
    import zmq.devices
    from zmq.devices import basedevice, monitoredqueue, monitoredqueuedevice