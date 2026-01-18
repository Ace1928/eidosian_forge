from . import errors
from . import revision as _mod_revision
from .branch import Branch
from .errors import BoundBranchOutOfDate
def remove_tags(branch, graph, old_tip, parents):
    """Remove tags on revisions between old_tip and new_tip.

    :param branch: Branch to remove tags from
    :param graph: Graph object for branch repository
    :param old_tip: Old branch tip
    :param parents: New parents
    :return: Names of the removed tags
    """
    reverse_tags = branch.tags.get_reverse_tag_dict()
    ancestors = graph.find_unique_ancestors(old_tip, parents)
    removed_tags = []
    for revid, tags in reverse_tags.items():
        if revid not in ancestors:
            continue
        for tag in tags:
            branch.tags.delete_tag(tag)
            removed_tags.append(tag)
    return removed_tags