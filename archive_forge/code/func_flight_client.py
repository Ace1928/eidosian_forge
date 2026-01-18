import asyncio
import pytest
import pyarrow
@pytest.fixture(scope='module')
def flight_client():
    with ExampleServer() as server:
        with flight.connect(f'grpc://localhost:{server.port}') as client:
            yield client