import socket
from typing import Iterable
from typing import List
from typing import Optional
from typing import Union
from selenium.types import AnyKey
from selenium.webdriver.common.keys import Keys
def is_connectable(port: int, host: Optional[str]='localhost') -> bool:
    """Tries to connect to the server at port to see if it is running.

    :Args:
     - port - The port to connect.
    """
    socket_ = None
    try:
        socket_ = socket.create_connection((host, port), 1)
        result = True
    except _is_connectable_exceptions:
        result = False
    finally:
        if socket_:
            socket_.close()
    return result