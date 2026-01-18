from asyncio import Future
import pytest
from jeepney.routing import Router
from jeepney.wrappers import new_method_return, new_error, DBusErrorResponse
from jeepney.bus_messages import message_bus
def test_message_reply():
    router = Router(Future)
    call = message_bus.Hello()
    future = router.outgoing(call)
    router.incoming(new_method_return(call, 's', ('test',)))
    assert future.result() == ('test',)