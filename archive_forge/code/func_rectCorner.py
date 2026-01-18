def rectCorner(x, y, width, height, anchor='sw', dims=False):
    """given rectangle controlled by x,y width and height return 
    the corner corresponding to the anchor"""
    if anchor not in ('nw', 'w', 'sw'):
        if anchor in ('n', 'c', 's'):
            x += width / 2.0
        else:
            x += width
    if anchor not in ('sw', 's', 'se'):
        if anchor in ('w', 'c', 'e'):
            y += height / 2.0
        else:
            y += height
    return (x, y, width, height) if dims else (x, y)