import sys
from operator import itemgetter
import click
from celery.bin.base import CeleryCommand, handle_preload_options
from celery.utils.graph import DependencyGraph, GraphFormatter
@graph.command(cls=CeleryCommand, context_settings={'allow_extra_args': True})
@click.pass_context
def bootsteps(ctx):
    """Display bootsteps graph."""
    worker = ctx.obj.app.WorkController()
    include = {arg.lower() for arg in ctx.args or ['worker', 'consumer']}
    if 'worker' in include:
        worker_graph = worker.blueprint.graph
        if 'consumer' in include:
            worker.blueprint.connect_with(worker.consumer.blueprint)
    else:
        worker_graph = worker.consumer.blueprint.graph
    worker_graph.to_dot(sys.stdout)