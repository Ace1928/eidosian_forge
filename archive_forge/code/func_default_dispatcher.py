from contextlib import contextmanager
from kombu.utils.objects import cached_property
@contextmanager
def default_dispatcher(self, hostname=None, enabled=True, buffer_while_offline=False):
    with self.app.amqp.producer_pool.acquire(block=True) as prod:
        with self.Dispatcher(prod.connection, hostname, enabled, prod.channel, buffer_while_offline) as d:
            yield d