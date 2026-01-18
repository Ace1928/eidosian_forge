def ps2tt(psfn):
    """ps fontname to family name, bold, italic"""
    psfn = psfn.lower()
    if psfn in _ps2tt_map:
        return _ps2tt_map[psfn]
    raise ValueError("Can't map determine family/bold/italic for %s" % psfn)