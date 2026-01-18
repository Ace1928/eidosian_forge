from typing import NamedTuple
def fake_client_factory():

    class FakeClient:
        """Fake AsyncHTTPClient

        body can be set in the test to a custom value.
        """
        body = b''

        async def fetch(*args, **kwargs):
            return Response(FakeClient.body)
    return FakeClient