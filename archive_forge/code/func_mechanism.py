import socket
import warnings
from io import BytesIO
from amqp.serialization import _write_table
@property
def mechanism(self):
    """Return a bytes containing the SASL mechanism name."""
    raise NotImplementedError