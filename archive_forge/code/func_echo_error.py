import pprint
import click
from amqp import Connection, Message
from click_repl import register_repl
from celery.bin.base import handle_preload_options
def echo_error(self, exception):
    self.cli_context.error(f'{self.cli_context.ERROR}: {exception}')