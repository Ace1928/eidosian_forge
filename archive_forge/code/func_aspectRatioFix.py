def aspectRatioFix(preserve, anchor, x, y, width, height, imWidth, imHeight, anchorAtXY=False):
    """This function helps position an image within a box.

    It first normalizes for two cases:
    - if the width is None, it assumes imWidth
    - ditto for height
    - if width or height is negative, it adjusts x or y and makes them positive

    Given
    (a) the enclosing box (defined by x,y,width,height where x,y is the         lower left corner) which you wish to position the image in, and
    (b) the image size (imWidth, imHeight), and
    (c) the 'anchor point' as a point of the compass - n,s,e,w,ne,se etc         and c for centre,

    this should return the position at which the image should be drawn,
    as well as a scale factor indicating what scaling has happened.

    It returns the parameters which would be used to draw the image
    without any adjustments:

        x,y, width, height, scale

    used in canvas.drawImage and drawInlineImage
    """
    scale = 1.0
    if width is None:
        width = imWidth
    if height is None:
        height = imHeight
    if width < 0:
        width = -width
        x -= width
    if height < 0:
        height = -height
        y -= height
    if preserve:
        imWidth = abs(imWidth)
        imHeight = abs(imHeight)
        scale = min(width / float(imWidth), height / float(imHeight))
        owidth = width
        oheight = height
        width = scale * imWidth - 1e-08
        height = scale * imHeight - 1e-08
        if not anchorAtXY:
            x, y = rectCorner(x, y, owidth - width, oheight - height, anchor)
    if anchorAtXY:
        if anchor not in ('sw', 's', 'se'):
            y -= height / 2.0 if anchor in ('e', 'c', 'w') else height
        if anchor not in ('nw', 'w', 'sw'):
            x -= width / 2.0 if anchor in ('n', 'c', 's') else width
    return (x, y, width, height, scale)