from . import errors
from . import graph as _mod_graph
from . import revision as _mod_revision
def merge_sort(graph, branch_tip, mainline_revisions=None, generate_revno=False):
    """Topological sort a graph which groups merges.

    :param graph: sequence of pairs of node->parents_list.
    :param branch_tip: the tip of the branch to graph. Revisions not
                       reachable from branch_tip are not included in the
                       output.
    :param mainline_revisions: If not None this forces a mainline to be
                               used rather than synthesised from the graph.
                               This must be a valid path through some part
                               of the graph. If the mainline does not cover all
                               the revisions, output stops at the start of the
                               old revision listed in the mainline revisions
                               list.
                               The order for this parameter is oldest-first.
    :param generate_revno: Optional parameter controlling the generation of
        revision number sequences in the output. See the output description of
        the MergeSorter docstring for details.
    :result: See the MergeSorter docstring for details.

    Node identifiers can be any hashable object, and are typically strings.
    """
    return MergeSorter(graph, branch_tip, mainline_revisions, generate_revno).sorted()