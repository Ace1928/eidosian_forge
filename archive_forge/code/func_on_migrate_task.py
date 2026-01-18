import click
from kombu import Connection
from celery.bin.base import CeleryCommand, CeleryOption, handle_preload_options
from celery.contrib.migrate import migrate_tasks
def on_migrate_task(state, body, message):
    ctx.obj.echo(f'Migrating task {state.count}/{state.strtotal}: {body}')