from operator import itemgetter
from breezy import controldir
from ... import errors, osutils, transport
from ...trace import note, show_error
from .helpers import best_format_for_objects_in_a_repository, single_plural
def dir_under_current(name):
    repo_base = self.repo.controldir.transport.base
    return osutils.pathjoin(repo_base, '..', name)