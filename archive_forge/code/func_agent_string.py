from io import BytesIO
from os import SEEK_END
import dulwich
from .errors import GitProtocolError, HangupException
def agent_string():
    return ('dulwich/' + '.'.join(map(str, dulwich.__version__))).encode('ascii')