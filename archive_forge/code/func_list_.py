import click
from celery.bin.base import CeleryCommand, handle_preload_options
@click.group(name='list')
@click.pass_context
@handle_preload_options
def list_(ctx):
    """Get info from broker.

    Note:

        For RabbitMQ the management plugin is required.
    """