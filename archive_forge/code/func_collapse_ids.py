import operator
from ... import (branch, commands, config, errors, option, trace, tsort, ui,
from ...revision import NULL_REVISION
from .classify import classify_delta
def collapse_ids(old_id, new_id, new_combos):
    old_combos = id_to_combos.pop(old_id)
    new_combos.update(old_combos)
    for old_user, old_email in old_combos:
        if old_user and old_user != user:
            low_old_user = old_user.lower()
            old_user_id = username_to_id[low_old_user]
            assert old_user_id in (old_id, new_id)
            username_to_id[low_old_user] = new_id
        if old_email and old_email != email:
            old_email_id = email_to_id[old_email]
            assert old_email_id in (old_id, new_id)
            email_to_id[old_email] = cur_id