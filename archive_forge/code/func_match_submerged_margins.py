import logging
import numpy as np
from matplotlib import _api, artist as martist
import matplotlib.transforms as mtransforms
import matplotlib._layoutgrid as mlayoutgrid
def match_submerged_margins(layoutgrids, fig):
    """
    Make the margins that are submerged inside an Axes the same size.

    This allows axes that span two columns (or rows) that are offset
    from one another to have the same size.

    This gives the proper layout for something like::
        fig = plt.figure(constrained_layout=True)
        axs = fig.subplot_mosaic("AAAB
CCDD")

    Without this routine, the axes D will be wider than C, because the
    margin width between the two columns in C has no width by default,
    whereas the margins between the two columns of D are set by the
    width of the margin between A and B. However, obviously the user would
    like C and D to be the same size, so we need to add constraints to these
    "submerged" margins.

    This routine makes all the interior margins the same, and the spacing
    between the three columns in A and the two column in C are all set to the
    margins between the two columns of D.

    See test_constrained_layout::test_constrained_layout12 for an example.
    """
    for sfig in fig.subfigs:
        match_submerged_margins(layoutgrids, sfig)
    axs = [a for a in fig.get_axes() if a.get_subplotspec() is not None and a.get_in_layout()]
    for ax1 in axs:
        ss1 = ax1.get_subplotspec()
        if ss1.get_gridspec() not in layoutgrids:
            axs.remove(ax1)
            continue
        lg1 = layoutgrids[ss1.get_gridspec()]
        if len(ss1.colspan) > 1:
            maxsubl = np.max(lg1.margin_vals['left'][ss1.colspan[1:]] + lg1.margin_vals['leftcb'][ss1.colspan[1:]])
            maxsubr = np.max(lg1.margin_vals['right'][ss1.colspan[:-1]] + lg1.margin_vals['rightcb'][ss1.colspan[:-1]])
            for ax2 in axs:
                ss2 = ax2.get_subplotspec()
                lg2 = layoutgrids[ss2.get_gridspec()]
                if lg2 is not None and len(ss2.colspan) > 1:
                    maxsubl2 = np.max(lg2.margin_vals['left'][ss2.colspan[1:]] + lg2.margin_vals['leftcb'][ss2.colspan[1:]])
                    if maxsubl2 > maxsubl:
                        maxsubl = maxsubl2
                    maxsubr2 = np.max(lg2.margin_vals['right'][ss2.colspan[:-1]] + lg2.margin_vals['rightcb'][ss2.colspan[:-1]])
                    if maxsubr2 > maxsubr:
                        maxsubr = maxsubr2
            for i in ss1.colspan[1:]:
                lg1.edit_margin_min('left', maxsubl, cell=i)
            for i in ss1.colspan[:-1]:
                lg1.edit_margin_min('right', maxsubr, cell=i)
        if len(ss1.rowspan) > 1:
            maxsubt = np.max(lg1.margin_vals['top'][ss1.rowspan[1:]] + lg1.margin_vals['topcb'][ss1.rowspan[1:]])
            maxsubb = np.max(lg1.margin_vals['bottom'][ss1.rowspan[:-1]] + lg1.margin_vals['bottomcb'][ss1.rowspan[:-1]])
            for ax2 in axs:
                ss2 = ax2.get_subplotspec()
                lg2 = layoutgrids[ss2.get_gridspec()]
                if lg2 is not None:
                    if len(ss2.rowspan) > 1:
                        maxsubt = np.max([np.max(lg2.margin_vals['top'][ss2.rowspan[1:]] + lg2.margin_vals['topcb'][ss2.rowspan[1:]]), maxsubt])
                        maxsubb = np.max([np.max(lg2.margin_vals['bottom'][ss2.rowspan[:-1]] + lg2.margin_vals['bottomcb'][ss2.rowspan[:-1]]), maxsubb])
            for i in ss1.rowspan[1:]:
                lg1.edit_margin_min('top', maxsubt, cell=i)
            for i in ss1.rowspan[:-1]:
                lg1.edit_margin_min('bottom', maxsubb, cell=i)