import asyncio
import pytest
import pyarrow
def test_get_flight_info_error(async_client):

    async def _test():
        descriptor = flight.FlightDescriptor.for_command(b'unknown')
        with pytest.raises(NotImplementedError) as excinfo:
            await async_client.get_flight_info(descriptor)
        assert 'Unknown command' in repr(excinfo.value)
    asyncio.run(_test())