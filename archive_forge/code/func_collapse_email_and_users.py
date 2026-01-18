import operator
from ... import (branch, commands, config, errors, option, trace, tsort, ui,
from ...revision import NULL_REVISION
from .classify import classify_delta
def collapse_email_and_users(email_users, combo_count):
    """Combine the mapping of User Name to email and email to User Name.

    If a given User Name is used for multiple emails, try to map it all to one
    entry.
    """
    id_to_combos = {}
    username_to_id = {}
    email_to_id = {}
    id_counter = 0

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
    for email, usernames in email_users.items():
        assert email not in email_to_id
        if not email:
            for user in usernames:
                if not user:
                    continue
                low_user = user.lower()
                user_id = username_to_id.get(low_user)
                if user_id is None:
                    id_counter += 1
                    user_id = id_counter
                    username_to_id[low_user] = user_id
                    id_to_combos[user_id] = id_combos = set()
                else:
                    id_combos = id_to_combos[user_id]
                id_combos.add((user, email))
            continue
        id_counter += 1
        cur_id = id_counter
        id_to_combos[cur_id] = id_combos = set()
        email_to_id[email] = cur_id
        for user in usernames:
            combo = (user, email)
            id_combos.add(combo)
            if not user:
                continue
            low_user = user.lower()
            user_id = username_to_id.get(low_user)
            if user_id is not None:
                if user_id != cur_id:
                    collapse_ids(user_id, cur_id, id_combos)
            username_to_id[low_user] = cur_id
    combo_to_best_combo = {}
    for cur_id, combos in id_to_combos.items():
        best_combo = sorted(combos, key=lambda x: combo_count[x], reverse=True)[0]
        for combo in combos:
            combo_to_best_combo[combo] = best_combo
    return combo_to_best_combo