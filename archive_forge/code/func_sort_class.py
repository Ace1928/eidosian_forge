import operator
from ... import (branch, commands, config, errors, option, trace, tsort, ui,
from ...revision import NULL_REVISION
from .classify import classify_delta
def sort_class(name):
    return [author for author, _ in sorted(ret[name].items(), key=classify_key)]