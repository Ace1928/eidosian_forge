import difflib
import patiencediff
from merge3 import Merge3
from ... import debug, merge, osutils
from ...trace import mutter
def merge_entries(base_entries, this_entries, other_entries, guess_edits=default_guess_edits):
    """Merge changelog given base, this, and other versions."""
    m3 = Merge3(base_entries, this_entries, other_entries, sequence_matcher=patiencediff.PatienceSequenceMatcher)
    result_entries = []
    at_top = True
    for group in m3.merge_groups():
        if 'changelog_merge' in debug.debug_flags:
            mutter('merge group:\n%r', group)
        group_kind = group[0]
        if group_kind == 'conflict':
            _, base, this, other = group
            new_in_other = [entry for entry in other if entry not in base]
            deleted_in_other = [entry for entry in base if entry not in other]
            if at_top and deleted_in_other:
                new_in_other, deleted_in_other, edits_in_other = guess_edits(new_in_other, deleted_in_other)
            else:
                edits_in_other = []
            if 'changelog_merge' in debug.debug_flags:
                mutter('at_top: %r', at_top)
                mutter('new_in_other: %r', new_in_other)
                mutter('deleted_in_other: %r', deleted_in_other)
                mutter('edits_in_other: %r', edits_in_other)
            updated_this = [entry for entry in this if entry not in deleted_in_other]
            for old_entry, new_entry in edits_in_other:
                try:
                    index = updated_this.index(old_entry)
                except ValueError:
                    raise EntryConflict()
                updated_this[index] = new_entry
            if 'changelog_merge' in debug.debug_flags:
                mutter('updated_this: %r', updated_this)
            if at_top:
                result_entries = new_in_other + result_entries
            else:
                result_entries.extend(new_in_other)
            result_entries.extend(updated_this)
        else:
            lines = group[1]
            result_entries.extend(lines)
        at_top = False
    return result_entries