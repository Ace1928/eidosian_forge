import asyncio
import pytest
import pyarrow
def test_get_flight_info(async_client):

    async def _test():
        descriptor = flight.FlightDescriptor.for_command(b'simple')
        info = await async_client.get_flight_info(descriptor)
        assert info == ExampleServer.simple_info
    asyncio.run(_test())