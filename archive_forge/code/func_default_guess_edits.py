import difflib
import patiencediff
from merge3 import Merge3
from ... import debug, merge, osutils
from ...trace import mutter
def default_guess_edits(new_entries, deleted_entries, entry_as_str=b''.join):
    """Default implementation of guess_edits param of merge_entries.

    This algorithm does O(N^2 * logN) SequenceMatcher.ratio() calls, which is
    pretty bad, but it shouldn't be used very often.
    """
    deleted_entries_as_strs = list(map(entry_as_str, deleted_entries))
    new_entries_as_strs = list(map(entry_as_str, new_entries))
    result_new = list(new_entries)
    result_deleted = list(deleted_entries)
    result_edits = []
    sm = difflib.SequenceMatcher()
    CUTOFF = 0.8
    while True:
        best = None
        best_score = CUTOFF
        for new_entry_as_str in new_entries_as_strs:
            sm.set_seq1(new_entry_as_str)
            for old_entry_as_str in deleted_entries_as_strs:
                sm.set_seq2(old_entry_as_str)
                score = sm.ratio()
                if score > best_score:
                    best = (new_entry_as_str, old_entry_as_str)
                    best_score = score
        if best is not None:
            del_index = deleted_entries_as_strs.index(best[1])
            new_index = new_entries_as_strs.index(best[0])
            result_edits.append((result_deleted[del_index], result_new[new_index]))
            del deleted_entries_as_strs[del_index], result_deleted[del_index]
            del new_entries_as_strs[new_index], result_new[new_index]
        else:
            break
    return (result_new, result_deleted, result_edits)