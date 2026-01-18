from io import BytesIO
from os import SEEK_END
import dulwich
from .errors import GitProtocolError, HangupException
def capability_agent():
    return CAPABILITY_AGENT + b'=' + agent_string()