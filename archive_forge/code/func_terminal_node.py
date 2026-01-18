import sys
from operator import itemgetter
import click
from celery.bin.base import CeleryCommand, handle_preload_options
from celery.utils.graph import DependencyGraph, GraphFormatter
def terminal_node(self, obj):
    return self.draw_node(obj, dict(self.term_scheme, **obj.scheme))