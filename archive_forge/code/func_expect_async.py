import asyncio
import errno
import signal
from pexpect import EOF
@asyncio.coroutine
def expect_async(expecter, timeout=None):
    idx = expecter.existing_data()
    if idx is not None:
        return idx
    if not expecter.spawn.async_pw_transport:
        pw = PatternWaiter()
        pw.set_expecter(expecter)
        transport, pw = (yield from asyncio.get_event_loop().connect_read_pipe(lambda: pw, expecter.spawn))
        expecter.spawn.async_pw_transport = (pw, transport)
    else:
        pw, transport = expecter.spawn.async_pw_transport
        pw.set_expecter(expecter)
        transport.resume_reading()
    try:
        return (yield from asyncio.wait_for(pw.fut, timeout))
    except asyncio.TimeoutError as e:
        transport.pause_reading()
        return expecter.timeout(e)