import io
import os
import re
import tempfile
import uuid
from contextlib import contextmanager
from cProfile import Profile
from functools import wraps
from ..config import config
from ..util import escape
from .state import state
def render_memray(name, sessions, show_memory_leaks=True, merge_threads=True, reporter='tree'):
    from memray import FileReader
    from memray.reporters.flamegraph import FlameGraphReporter
    from memray.reporters.stats import StatsReporter
    from memray.reporters.table import TableReporter
    from memray.reporters.tree import TreeReporter
    reporter_cls = {'flamegraph': FlameGraphReporter, 'stats': StatsReporter, 'table': TableReporter, 'tree': TreeReporter}.get(reporter)
    session = sessions[-1]
    with tempfile.NamedTemporaryFile() as nf:
        nf.write(session)
        nf.flush()
        reader = FileReader(nf.name)
        if show_memory_leaks:
            snapshot = reader.get_leaked_allocation_records(merge_threads=merge_threads if merge_threads is not None else True)
        else:
            snapshot = reader.get_high_watermark_allocation_records(merge_threads=merge_threads if merge_threads is not None else True)
        kwargs = {'native_traces': reader.metadata.has_native_traces}
        if reporter in ('flamegraph', 'table'):
            kwargs['memory_records'] = tuple(reader.get_memory_snapshots())
        reporter_inst = reporter_cls.from_snapshot(snapshot, **kwargs)
    out = io.StringIO()
    if reporter == 'flamegraph':
        reporter_inst.render(out, reader.metadata, show_memory_leaks, merge_threads)
    elif reporter == 'table':
        reporter_inst.render(out, reader.metadata, show_memory_leaks)
    else:
        reporter_inst.render(file=out)
    out.seek(0)
    return (out.read(), '')