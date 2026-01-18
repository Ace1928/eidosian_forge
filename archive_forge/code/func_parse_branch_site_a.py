import re
def parse_branch_site_a(foreground, line_floats, site_classes):
    """Parse results specific to the branch site A model."""
    if not site_classes or len(line_floats) == 0:
        return
    for n in range(len(line_floats)):
        if site_classes[n].get('branch types') is None:
            site_classes[n]['branch types'] = {}
        if foreground:
            site_classes[n]['branch types']['foreground'] = line_floats[n]
        else:
            site_classes[n]['branch types']['background'] = line_floats[n]
    return site_classes