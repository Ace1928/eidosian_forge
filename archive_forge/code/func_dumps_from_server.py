import io
import ray
from typing import Any
from typing import TYPE_CHECKING
from ray._private.client_mode_hook import disable_client_hook
import ray.cloudpickle as cloudpickle
from ray.util.client.client_pickler import PickleStub
from ray.util.client.server.server_stubs import ClientReferenceActor
from ray.util.client.server.server_stubs import ClientReferenceFunction
import pickle  # noqa: F401
def dumps_from_server(obj: Any, client_id: str, server_instance: 'RayletServicer', protocol=None) -> bytes:
    with io.BytesIO() as file:
        sp = ServerPickler(client_id, server_instance, file, protocol=protocol)
        sp.dump(obj)
        return file.getvalue()