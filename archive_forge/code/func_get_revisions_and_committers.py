import operator
from ... import (branch, commands, config, errors, option, trace, tsort, ui,
from ...revision import NULL_REVISION
from .classify import classify_delta
def get_revisions_and_committers(a_repo, revids):
    """Get the Revision information, and the best-match for committer."""
    email_users = {}
    combo_count = {}
    with ui.ui_factory.nested_progress_bar() as pb:
        trace.note('getting revisions')
        revisions = list(a_repo.iter_revisions(revids))
        for count, (revid, rev) in enumerate(revisions):
            pb.update('checking', count, len(revids))
            for author in rev.get_apparent_authors():
                username, email = config.parse_username(author)
                email_users.setdefault(email, set()).add(username)
                combo = (username, email)
                combo_count[combo] = combo_count.setdefault(combo, 0) + 1
    return ((rev for revid, rev in revisions), collapse_email_and_users(email_users, combo_count))