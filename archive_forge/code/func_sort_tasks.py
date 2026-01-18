import datetime
import time
from .search import parse_search_terms, satisfies_search_terms
def sort_tasks(tasks, sort_by):
    assert sort_by.lstrip('-') in sort_keys
    reverse = False
    if sort_by.startswith('-'):
        sort_by = sort_by.lstrip('-')
        reverse = True
    for task in sorted(tasks, key=lambda x: getattr(x[1], sort_by) or sort_keys[sort_by](), reverse=reverse):
        yield task