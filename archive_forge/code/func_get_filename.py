import breezy
import breezy.commands
import breezy.help
from breezy.doc_generate import get_autodoc_datetime
def get_filename(options):
    return '%s.bash_completion' % options.brz_name