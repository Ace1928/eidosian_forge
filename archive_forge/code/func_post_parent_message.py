from __future__ import annotations
from streamlit.proto.ForwardMsg_pb2 import ForwardMsg
from streamlit.runtime.scriptrunner import get_script_run_ctx
def post_parent_message(message: str) -> None:
    """
    Sends a string message to the parent window (when host configuration allows).
    """
    ctx = get_script_run_ctx()
    if ctx is None:
        return
    fwd_msg = ForwardMsg()
    fwd_msg.parent_message.message = message
    ctx.enqueue(fwd_msg)