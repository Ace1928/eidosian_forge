import pprint
import click
from amqp import Connection, Message
from click_repl import register_repl
from celery.bin.base import handle_preload_options
def echo_ok(self):
    self.cli_context.echo(self.cli_context.OK)