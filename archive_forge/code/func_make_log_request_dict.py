import codecs
import itertools
import re
import sys
from io import BytesIO
from typing import Callable, Dict, List
from warnings import warn
from .lazy_import import lazy_import
from breezy import (
from breezy.i18n import gettext, ngettext
from . import errors, registry
from . import revision as _mod_revision
from . import revisionspec, trace
from . import transport as _mod_transport
from .osutils import (format_date,
from .tree import InterTree, find_previous_path
def make_log_request_dict(direction='reverse', specific_files=None, start_revision=None, end_revision=None, limit=None, message_search=None, levels=None, generate_tags=True, delta_type=None, diff_type=None, _match_using_deltas=True, exclude_common_ancestry=False, match=None, signature=False, omit_merges=False):
    """Convenience function for making a logging request dictionary.

    Using this function may make code slightly safer by ensuring
    parameters have the correct names. It also provides a reference
    point for documenting the supported parameters.

    :param direction: 'reverse' (default) is latest to earliest;
      'forward' is earliest to latest.

    :param specific_files: If not None, only include revisions
      affecting the specified files, rather than all revisions.

    :param start_revision: If not None, only generate
      revisions >= start_revision

    :param end_revision: If not None, only generate
      revisions <= end_revision

    :param limit: If set, generate only 'limit' revisions, all revisions
      are shown if None or 0.

    :param message_search: If not None, only include revisions with
      matching commit messages

    :param levels: the number of levels of revisions to
      generate; 1 for just the mainline; 0 for all levels, or None for
      a sensible default.

    :param generate_tags: If True, include tags for matched revisions.
`
    :param delta_type: Either 'full', 'partial' or None.
      'full' means generate the complete delta - adds/deletes/modifies/etc;
      'partial' means filter the delta using specific_files;
      None means do not generate any delta.

    :param diff_type: Either 'full', 'partial' or None.
      'full' means generate the complete diff - adds/deletes/modifies/etc;
      'partial' means filter the diff using specific_files;
      None means do not generate any diff.

    :param _match_using_deltas: a private parameter controlling the
      algorithm used for matching specific_files. This parameter
      may be removed in the future so breezy client code should NOT
      use it.

    :param exclude_common_ancestry: Whether -rX..Y should be interpreted as a
      range operator or as a graph difference.

    :param signature: show digital signature information

    :param match: Dictionary of list of search strings to use when filtering
      revisions. Keys can be 'message', 'author', 'committer', 'bugs' or
      the empty string to match any of the preceding properties.

    :param omit_merges: If True, commits with more than one parent are
      omitted.

    """
    if message_search:
        if match:
            if 'message' in match:
                match['message'].append(message_search)
            else:
                match['message'] = [message_search]
        else:
            match = {'message': [message_search]}
    return {'direction': direction, 'specific_files': specific_files, 'start_revision': start_revision, 'end_revision': end_revision, 'limit': limit, 'levels': levels, 'generate_tags': generate_tags, 'delta_type': delta_type, 'diff_type': diff_type, 'exclude_common_ancestry': exclude_common_ancestry, 'signature': signature, 'match': match, 'omit_merges': omit_merges, '_match_using_deltas': _match_using_deltas}