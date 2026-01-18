from tempfile import TemporaryFile
import threading
import pytest
from jeepney import (
from jeepney.io.threading import open_dbus_connection, DBusRouter, Proxy
@pytest.fixture()
def respond_with_fd():
    name = 'io.gitlab.takluyver.jeepney.tests.respond_with_fd'
    addr = DBusAddress(bus_name=name, object_path='/')
    with open_dbus_connection(bus='SESSION', enable_fds=True) as conn:
        with DBusRouter(conn) as router:
            status, = Proxy(message_bus, router).RequestName(name)
        assert status == 1

        def _reply_once():
            while True:
                msg = conn.receive()
                if msg.header.message_type is MessageType.method_call:
                    if msg.header.fields[HeaderFields.member] == 'GetFD':
                        with TemporaryFile('w+') as tf:
                            tf.write('readme')
                            tf.seek(0)
                            rep = new_method_return(msg, 'h', (tf,))
                            conn.send(rep)
                            return
                    else:
                        conn.send(new_error(msg, 'NoMethod'))
        reply_thread = threading.Thread(target=_reply_once, daemon=True)
        reply_thread.start()
        yield addr
    reply_thread.join()