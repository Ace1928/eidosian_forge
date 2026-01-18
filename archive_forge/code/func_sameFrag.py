import reportlab
def sameFrag(f, g):
    """returns 1 if two ParaFrags map out the same"""
    if hasattr(f, 'cbDefn') or hasattr(g, 'cbDefn') or hasattr(f, 'lineBreak') or hasattr(g, 'lineBreak'):
        return 0
    for a in ('fontName', 'fontSize', 'textColor', 'rise', 'us_lines', 'link', 'backColor', 'nobr'):
        if getattr(f, a, None) != getattr(g, a, None):
            return 0
    return 1