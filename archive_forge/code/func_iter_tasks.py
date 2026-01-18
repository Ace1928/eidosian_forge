import datetime
import time
from .search import parse_search_terms, satisfies_search_terms
def iter_tasks(events, limit=None, offset=0, type=None, worker=None, state=None, sort_by=None, received_start=None, received_end=None, started_start=None, started_end=None, search=None):
    i = 0
    tasks = events.state.tasks_by_timestamp()
    if sort_by is not None:
        tasks = sort_tasks(tasks, sort_by)

    def convert(x):
        return time.mktime(datetime.datetime.strptime(x, '%Y-%m-%d %H:%M').timetuple())
    search_terms = parse_search_terms(search or {})
    for uuid, task in tasks:
        if type and task.name != type:
            continue
        if worker and task.worker and (task.worker.hostname != worker):
            continue
        if state and task.state != state:
            continue
        if received_start and task.received and (task.received < convert(received_start)):
            continue
        if received_end and task.received and (task.received > convert(received_end)):
            continue
        if started_start and task.started and (task.started < convert(started_start)):
            continue
        if started_end and task.started and (task.started > convert(started_end)):
            continue
        if not satisfies_search_terms(task, search_terms):
            continue
        if i >= offset:
            yield (uuid, task)
        i += 1
        if limit is not None:
            if i == limit + offset:
                break